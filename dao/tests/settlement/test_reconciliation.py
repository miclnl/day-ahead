"""E1a contracts: required observations, detached provenance and canonical identity."""

import ast
import inspect
import json
from dataclasses import MISSING, asdict, fields, is_dataclass, replace
from datetime import date, datetime
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import get_args, get_type_hints

import pytest

from dao.prog.settlement import reconciliation as r
from dao.prog.settlement.repository import _json
from dao.prog.settlement.types import (
    InputInterval,
    MonthState,
    PeriodAdjustment,
    SettlementLine,
    SettlementState,
    SettlementView,
    SpotInterval,
)
from dao.tests.settlement.conftest import START_TS
from dao.tests.settlement.test_repository import METER, publish
from dao.tests.settlement.test_repository import calculate as calculate
from dao.tests.settlement.test_repository import inputs as inputs
from dao.tests.settlement.test_repository import repo as repo
from dao.tests.settlement.test_repository import terms as terms

DTO_NAMES = (
    "SupplierAmounts", "SupplierCredit", "SupplierRounding", "SupplierPeriod",
    "BookSourcePin", "RunSource", "LineSource", "CreditSource",
    "ReconciliationSources", "ComponentComparison", "ReconciliationReport",
    "ReconciliationBasis",
)


def e1b_rounding(**changes):
    return rounding(**{
        'method': 'period_component_half_up_v1', 'money_quantum': Decimal('0.01'),
        'money_mode': 'ROUND_HALF_UP', 'aggregation': 'sum_unrounded_then_round_each_component',
        'total_rule': 'sum_rounded_gross_components', 'energy_quantum': Decimal('0.001'),
        'energy_mode': 'ROUND_HALF_UP', 'evidence_ref': 'synthetic-rounding-v1', **changes,
    })


def e1b_amounts(gross='0', ex=None, vat='0'):
    return r.SupplierAmounts(Decimal(gross if ex is None else ex), Decimal(vat), Decimal(gross))


def e1b_hash_snapshot(snapshot):
    return replace(snapshot, snapshot_hash=r._digest({
        k: v for k, v in asdict(snapshot).items() if k != 'snapshot_hash'
    }))


def e1b_line_source(line, item, snapshot, generation=1, run_id='run-1'):
    digest = r._digest(line)
    return r.LineSource(
        r._canonical_json(['settlement-line-v1', 'm', line.start_ts, generation, run_id, digest]),
        line.start_ts, line.end_ts, line.input_revision, line.input_hash, line.snapshot_hash,
        generation, run_id, digest,
        r._digest({k: v for k, v in asdict(item).items() if k != 'input_revision'}),
        item, snapshot, (),
    )


@pytest.fixture
def e1b_case(snapshot):
    def build(*, spots=('0.10', '0.10'), imports=None, exports=None, gross=None,
              export_gross=None, bonus=None, terms=None, start=START_TS, quality='measured'):
        n = len(spots)
        imports = ('1',) * n if imports is None else imports
        exports = ('0',) * n if exports is None else exports
        terms = terms or {}
        tariff = e1b_hash_snapshot(snapshot(**{'rule_version': 'tariffs-v1', 'contract_id': 'c',
                                              'monthly_export_floor': 'none', **terms}))
        lines, sources = [], []
        for i, spot in enumerate(spots):
            at = start + i * 900
            item = InputInterval(
                start_ts=at, end_ts=at + 900, import_kwh=Decimal(imports[i]),
                export_kwh=Decimal(exports[i]), battery_discharge_kwh=Decimal('0'),
                origin='verified_pv', quality=quality, sources=('synthetic-meter',),
                spot_ex_vat=Decimal(spot), spot_source='synthetic-spot',
                input_hash=f'input-{i}', meter_id='m', input_revision=1,
            )
            with localcontext() as ctx:
                ctx.prec = 36
                imp = Decimal(gross[i]) if gross else Decimal(spot) * item.import_kwh
                exp = Decimal(export_gross[i]) if export_gross else Decimal(spot) * item.export_kwh
            line = SettlementLine(
                at, at + 900, 'm', 1, item.input_hash, tariff.snapshot_hash,
                imp, exp, Decimal(bonus[i]) if bonus else Decimal('0'), Decimal('0'),
                SettlementState(at, None, None), SettlementState(at + 900, None, None), quality, (),
            )
            lines.append(line)
            sources.append(e1b_line_source(line, item, tariff))
        with localcontext() as ctx:
            ctx.prec = 100
            net = sum((x.import_cost - x.base_export_revenue - x.bonus_revenue for x in lines), Decimal(0))
        view = SettlementView('m', 1, 1, 1, 'current', tuple(lines), (), net, net,
                              start, start + n * 900, True, start + n * 900, None, None)
        pin = r.BookSourcePin('m', 1, 1, 1, 1, None, start, start + n * 900, r._digest(view),
                              start + n * 900 + 1)
        run = r.RunSource('run-1', 1, 1, 'settlement-v1', 'whole-input', 'whole-result',
                          (tariff.snapshot_hash,))
        bundle = r.ReconciliationSources(pin, (run,), tuple(sources), (), ())
        supplied = supplier(start_ts=start, end_ts=start + n * 900,
                            import_kwh=Decimal('2'), export_kwh=Decimal('0'),
                            import_amounts=e1b_amounts('0.20'), base_export_amounts=e1b_amounts(),
                            bonus_amounts=e1b_amounts(), credits=(), total=Decimal('0.20'),
                            rounding=e1b_rounding())
        return view, supplied, bundle
    return build


def e1b_repin(view, sources):
    return replace(sources, pin=replace(sources.pin, view_digest=r._digest(view),
                                       generation=view.generation, view_input_revision=view.input_revision,
                                       current_input_revision=view.current_input_revision,
                                       dirty_from_ts=view.dirty_from_ts))


def e1b_credit(case, *, ex='0.04', vat='0.01', gross='0.05', quality='estimated', kind='spot_zero',
               supplier_amounts=None):
    view, supplied, sources = case
    start, end = view.coverage_start_ts, view.coverage_end_ts
    month = MonthState('c', start, end, Decimal('-0.04'), ((Decimal('25'), Decimal('0.04')),), True)
    line = replace(view.lines[-1], after_state=replace(view.lines[-1].after_state, month=month))
    last = e1b_line_source(line, sources.lines[-1].item, sources.lines[-1].snapshot)
    adjustment = PeriodAdjustment(kind, start, end, Decimal(ex), Decimal(vat), Decimal(gross),
                                  Decimal('0.04'), quality)
    digest = r._digest(adjustment)
    cs = r.CreditSource(r._canonical_json(['settlement-credit-v1', 'm', kind, start, end, 1,
                                          'run-1', digest]), kind, start, end, 1, 'run-1', digest,
                        'c', 'Europe/Amsterdam', line, last, ())
    with localcontext() as ctx:
        ctx.prec = 36
        net = view.net_cost - Decimal(gross)
    view = replace(view, lines=(*view.lines[:-1], line), adjustments=(adjustment,),
                   net_cost=net, known_subtotal=net)
    sources = e1b_repin(view, replace(sources, lines=(*sources.lines[:-1], last), credits=(cs,)))
    supplied = replace(supplied, credits=(credit(kind=kind, period_start_ts=start, period_end_ts=end,
                                               amounts=supplier_amounts or e1b_amounts(gross, ex, vat)),))
    return view, supplied, sources


def e1b_compare(case):
    view, supplied, sources = case
    return r.reconcile(view, supplied, sources=sources)


def e1b_rows(report):
    return {c.key: c for c in (*report.components, report.total)}


@pytest.mark.parametrize('quantum', ['1.000', '0.1000', '0.0010'])
@pytest.mark.parametrize('direction', ['import', 'export'])
@pytest.mark.parametrize('observation', ['1.000', '1.0004'])
def test_e1b_fix1_canonical_energy_quantum(e1b_case, quantum, direction, observation):
    view, supplied, sources = e1b_case(spots=('0.10',), imports=('1.0004',), exports=('1.0004',))
    supplied = replace(supplied, rounding=e1b_rounding(energy_quantum=Decimal(quantum)),
                       **{f'{direction}_kwh': Decimal(observation)})
    report = e1b_compare((view, supplied, sources))
    row = e1b_rows(report)[f'{direction}_kwh']
    assert row.book_value == Decimal('1.0004')
    assert row.supplier_value == Decimal(observation)
    assert report.supplier.rounding.energy_quantum.as_tuple() == Decimal(quantum).as_tuple()
    if observation == '1.000':
        assert row.comparable_book_value == Decimal('1')
        assert row.delta == 0 and row.status == 'matched'
    else:
        assert row.delta is None and row.status == 'not_comparable'
        assert 'supplier_quantum_mismatch' in row.reasons


@pytest.mark.parametrize('observation', ['0.10', '0.1004'])
def test_e1b_fix1_canonical_money_quantum(e1b_case, observation):
    view, supplied, sources = e1b_case(spots=('0.1004',))
    supplied = replace(supplied, rounding=e1b_rounding(money_quantum=Decimal('0.0100')),
                       import_amounts=e1b_amounts(observation))
    report = e1b_compare((view, supplied, sources))
    row = e1b_rows(report)['import.gross']
    assert row.book_value == Decimal('0.1004')
    assert report.supplier.rounding.money_quantum.as_tuple() == Decimal('0.0100').as_tuple()
    if observation == '0.10':
        assert row.comparable_book_value == Decimal('0.10') and row.delta == 0
    else:
        assert row.delta is None and 'supplier_quantum_mismatch' in row.reasons


@pytest.mark.parametrize('total,reason', [(None, 'supplier_missing'),
                                        (Decimal('0.21'), 'supplier_total_inconsistent')])
def test_e1b_fix1_null_total_is_not_inconsistent(e1b_case, total, reason):
    view, supplied, sources = e1b_case()
    report = e1b_compare((view, replace(supplied, total=total), sources))
    assert report.total.reasons == (reason,)
    assert report.total.supplier_value == total
    assert report.total.book_value == Decimal('0.20')
    assert report.total.delta is None and report.status == 'not_comparable'
    assert all(row.status == 'matched' for row in report.components)


def test_e1b_fix1_missing_net_never_uses_known_subtotal(e1b_case):
    view, supplied, sources = e1b_case()
    view = replace(view, net_cost=None, known_subtotal=Decimal('0.20'), complete=False)
    sources = e1b_repin(view, sources)
    report = e1b_compare((view, supplied, sources))
    assert view.known_subtotal == Decimal('0.20')
    assert report.total.book_value is None
    assert report.total.comparable_book_value is None and report.total.delta is None
    assert report.total.status == report.status == 'not_comparable'
    assert 'book_value_missing' in report.total.reasons


@pytest.mark.parametrize('precision', [6, 28, 50])
@pytest.mark.parametrize('energy,multiplier,vat_percent,ex,vat,gross,supplier_energy,supplier_ex,supplier_gross', [
    ('1.23456789', '1', '21',
     '0.15241578751672002468', '0.0320073153785112051828', '0.1844231028952312298628',
     '1.235', '0.15', '0.18'),
    ('1.123456789', '1.123456789012', '21.123456789012',
     '0.155821622888980439155063419281616', '0.0329149131788910151023289357301966576',
     '0.188736536067871454257392355011812658', '1.123', '0.16', '0.19'),
], ids=['above6', 'above28'])
def test_e1b_fix1_projection_context36_independent_pins(
    e1b_case, precision, energy, multiplier, vat_percent, ex, vat, gross,
    supplier_energy, supplier_ex, supplier_gross,
):
    # Literal pins from separate integer/Fraction arithmetic, not quote/reconcile or
    # Decimal arithmetic under the production/test ambient context. The second
    # VAT product and gross sum require rounding to exactly 36 significant digits.
    terms = {f'{name}_{direction}': ((date(2027, 1, 1), Decimal(value)),)
             for direction in ('consumption', 'production')
             for name, value in (('multiplier', multiplier), ('vat', vat_percent))}
    view, supplied, sources = e1b_case(spots=('0.123456789012',), imports=(energy,), exports=(energy,),
                                     gross=(gross,), export_gross=(gross,), terms=terms)
    amounts = e1b_amounts(supplier_gross, supplier_ex, '0.03')
    supplied = replace(supplied, import_amounts=amounts, base_export_amounts=amounts,
                       import_kwh=Decimal(supplier_energy), export_kwh=Decimal(supplier_energy),
                       total=Decimal('0'))
    with localcontext() as ctx:
        ctx.prec = precision
        report = e1b_compare((view, supplied, sources))
    rows = e1b_rows(report)
    for name in ('import', 'base_export'):
        assert rows[f'{name}.ex_vat'].book_value == Decimal(ex)
        assert rows[f'{name}.vat'].book_value == Decimal(vat)
        assert rows[f'{name}.gross'].book_value == Decimal(gross)
        assert rows[f'{name}.ex_vat'].comparable_book_value == Decimal(supplier_ex)
        assert rows[f'{name}.vat'].comparable_book_value == Decimal('0.03')
        assert rows[f'{name}.gross'].comparable_book_value == Decimal(supplier_gross)
    assert report.total.book_value == report.total.delta == Decimal('0')
    assert report.status == 'matched'
    assert report.provenance == sources


def test_e1b_exact_components_and_signs(e1b_case):
    case = e1b_case()
    report = e1b_compare(case)
    rows = e1b_rows(report)
    expected = {'import_kwh': '2', 'export_kwh': '0', 'import.gross': '0.20',
                'import.ex_vat': '0.20', 'import.vat': '0', 'net_cost': '0.20',
                **{f'{k}.{part}': '0' for k in ('base_export', 'bonus')
                   for part in ('gross', 'ex_vat', 'vat')}}
    assert set(rows) == set(expected)
    for key, value in expected.items():
        assert rows[key].book_value == Decimal(value)
        assert rows[key].delta == 0 and rows[key].status == 'matched'
    assert report.status == 'matched'
    assert report.provenance == case[2] and report.supplier == case[1]
    assert report.supplier_digest == r._digest(case[1])
    view, supplied, sources = case
    changed = e1b_compare((view, replace(supplied, import_amounts=e1b_amounts('0.21'),
                                       total=Decimal('0.21')), sources))
    assert e1b_rows(changed)['import.gross'].delta == changed.total.delta == Decimal('0.01')
    assert changed.status == 'different'
    case = e1b_credit(e1b_case(spots=('-0.20',), exports=('0.5',), bonus=('0.03',)))
    view, supplied, sources = case
    supplied = replace(supplied, import_kwh=Decimal(1), export_kwh=Decimal('0.5'),
                       import_amounts=e1b_amounts('-0.20'), base_export_amounts=e1b_amounts('-0.09'),
                       bonus_amounts=e1b_amounts('0.03'), total=Decimal('-0.19'))
    report = e1b_compare((view, supplied, sources))
    assert report.total.book_value == Decimal('-0.18')
    assert report.total.delta == Decimal('-0.01')
    assert e1b_rows(report)['base_export.gross'].delta == Decimal('0.01')
    assert report.status == 'not_comparable'


@pytest.mark.parametrize('spots,exports,export_gross,want,net', [
    (('0.005', '0.005'), None, None, '0.01', '0.010'),
    (('-0.005',), None, None, '-0.01', '-0.005'),
    (('0.005',), ('1',), ('0.004',), '0.01', '0.001'),
])
def test_e1b_sum_then_round_and_signed_half_up(e1b_case, spots, exports, export_gross, want, net):
    view, supplied, sources = e1b_case(spots=spots, exports=exports, export_gross=export_gross)
    supplied = replace(supplied, import_amounts=e1b_amounts(want), total=Decimal(want))
    report = e1b_compare((view, supplied, sources))
    assert e1b_rows(report)['import.gross'].comparable_book_value == Decimal(want)
    assert report.total.book_value == Decimal(net)
    assert report.total.comparable_book_value == Decimal(want)
    assert report.total.delta == 0


@pytest.mark.parametrize('field,value,reason', [
    ('method', 'other', 'rounding_unsupported'), ('money_quantum', Decimal('0.1'), 'rounding_unsupported'),
    ('money_mode', 'ROUND_HALF_EVEN', 'rounding_unsupported'),
    ('aggregation', 'per_interval', 'rounding_unsupported'),
    ('total_rule', 'round_net', 'rounding_unsupported'),
    ('energy_quantum', Decimal('0.03'), 'rounding_unsupported'),
    ('energy_mode', 'ROUND_DOWN', 'rounding_unsupported'),
    ('evidence_ref', '', 'rounding_missing'), ('evidence_ref', None, 'rounding_missing'),
])
def test_e1b_rounding_declaration_and_supplier_quantum_invalid(e1b_case, field, value, reason):
    view, supplied, sources = e1b_case()
    report = e1b_compare((view, replace(supplied, rounding=e1b_rounding(**{field: value})), sources))
    assert report.total.delta is None
    assert reason in report.total.reasons


@pytest.mark.parametrize('quantum', ['1', '0.1', '0.01', '0.001', '0.0001', '0.00001',
                                   '0.000001', '0.0000001', '0.00000001', '0.000000001'])
def test_e1b_rounding_declaration_and_supplier_quantum_allowed(e1b_case, quantum):
    view, supplied, sources = e1b_case()
    assert e1b_compare((view, replace(supplied, rounding=e1b_rounding(energy_quantum=Decimal(quantum))),
                        sources)).status == 'matched'


def test_e1b_rounding_declaration_and_supplier_quantum_missing_or_off_grid(e1b_case):
    view, supplied, sources = e1b_case()
    report = e1b_compare((view, replace(supplied, rounding=None), sources))
    assert 'rounding_missing' in report.total.reasons and report.total.delta is None
    report = e1b_compare((view, replace(supplied, import_amounts=e1b_amounts('0.005')), sources))
    row = e1b_rows(report)['import.gross']
    assert row.supplier_value == Decimal('0.005') and row.delta is None
    assert row.status == 'not_comparable'
    report = e1b_compare((view, replace(supplied, import_kwh=Decimal('2.0001')), sources))
    assert e1b_rows(report)['import_kwh'].delta is None


def test_e1b_supplier_total_and_compensating_deltas(e1b_case):
    view, supplied, sources = e1b_case()
    report = e1b_compare((view, replace(supplied, total=Decimal('0.21')), sources))
    assert report.total.delta is None and 'supplier_total_inconsistent' in report.total.reasons
    assert e1b_rows(report)['import.gross'].delta == 0
    report = e1b_compare((view, replace(supplied, import_amounts=e1b_amounts('0.21'),
                                      base_export_amounts=e1b_amounts('0.01')), sources))
    assert report.total.delta == 0 and report.status == 'different'
    for key in ('import.gross', 'base_export.gross'):
        assert e1b_rows(report)[key].delta == Decimal('0.01')


@pytest.mark.parametrize('precision', [6, 28, 50])
@pytest.mark.parametrize('gross,proven', [('0.4114', True), ('0.411400000001', False)])
def test_e1b_vat_projection_exact_or_unproven(e1b_case, precision, gross, proven):
    terms = {name: ((date(2027, 1, 1), Decimal(value)),) for name, value in (
        ('energy_taxes_consumption', '0.05'), ('cost_supplier_consumption', '0.02'),
        ('vat_consumption', '21'))}
    view, supplied, sources = e1b_case(spots=('0.10',), imports=('2',), exports=('1',),
                                     gross=(gross,), terms=terms)
    supplied = replace(supplied, export_kwh=Decimal(1), import_amounts=e1b_amounts('0.41', '0.34', '0.07'),
                       base_export_amounts=e1b_amounts('0.10'), total=Decimal('0.31'))
    with localcontext() as ctx:
        ctx.prec = precision
        report = e1b_compare((view, supplied, sources))
    rows = e1b_rows(report)
    assert rows['import.gross'].book_value == Decimal(gross)
    assert rows['import.gross'].delta == 0
    if proven:
        assert rows['import.ex_vat'].book_value == Decimal('0.34')
        assert rows['import.vat'].book_value == Decimal('0.0714')
        assert report.status == 'matched'
    else:
        for key in ('import.ex_vat', 'import.vat'):
            assert rows[key].delta is None
            assert 'component_projection_unproven' in rows[key].reasons


@pytest.mark.parametrize('mode', ['absent', 'disabled', 'enabled_zero', 'nonzero'])
def test_e1b_bonus_split_and_structural_zero(e1b_case, bonus_rules, mode):
    terms = {} if mode == 'absent' else {'bonus': replace(bonus_rules, enabled=mode != 'disabled')}
    case = e1b_case(bonus=('0.03', '0') if mode == 'nonzero' else None, terms=terms)
    view, supplied, sources = case
    supplied = replace(supplied, bonus_amounts=e1b_amounts('0.03' if mode == 'nonzero' else '0'),
                       total=Decimal('0.17' if mode == 'nonzero' else '0.20'))
    report = e1b_compare((view, supplied, sources))
    assert report.total.delta == 0
    assert e1b_rows(report)['bonus.gross'].delta == 0
    for key in ('bonus.ex_vat', 'bonus.vat'):
        row = e1b_rows(report)[key]
        if mode in ('absent', 'disabled'):
            assert row.book_value == 0 and row.delta == 0
        else:
            assert row.delta is None and 'bonus_vat_split_unavailable' in row.reasons
            assert report.status == 'not_comparable'


@pytest.mark.parametrize('change', ['matched', 'missing', 'extra', 'unknown_set', 'zero_missing',
                                  'contract', 'zone', 'period', 'kind', 'unproven', 'inconsistent'])
def test_e1b_credit_union_identity_and_estimated_quality(e1b_case, change):
    case = e1b_credit(e1b_case(), gross='0' if change == 'zero_missing' else '0.05',
                      ex='0' if change == 'zero_missing' else '0.04',
                      vat='0' if change == 'zero_missing' else '0.01',
                      kind='other' if change == 'kind' else 'spot_zero')
    view, supplied, sources = case
    supplied = replace(supplied, total=Decimal('0.15'))
    if change in ('missing', 'zero_missing'):
        supplied = replace(supplied, credits=())
    elif change == 'extra':
        supplied = replace(supplied, credits=(*supplied.credits,
                           credit(period_start_ts=START_TS - 900, period_end_ts=START_TS,
                                  amounts=e1b_amounts())))
    elif change == 'unknown_set':
        supplied = replace(supplied, credits=None)
    elif change == 'contract':
        supplied = replace(supplied, contract_id='different')
    elif change == 'zone':
        supplied = replace(supplied, timezone='UTC')
    elif change == 'period':
        supplied = replace(supplied, credits=(replace(supplied.credits[0], period_start_ts=START_TS - 900),))
    elif change == 'unproven':
        sources = replace(sources, credits=(replace(sources.credits[0], closing_source=None),))
    elif change == 'inconsistent':
        view, supplied, sources = e1b_credit(e1b_case(), gross='0.06')
    report = e1b_compare((view, supplied, sources))
    rows = [x for x in report.components if x.key.startswith('credit:')]
    assert rows and report.total.book_value == view.net_cost
    if change == 'matched':
        assert all(x.status == 'matched' and x.quality == 'estimated' for x in rows)
        assert report.total.quality == 'estimated' and report.total.delta == 0
        assert report.status == 'matched'
    else:
        assert report.status == 'not_comparable' and report.total.delta is None
        assert any(x.status == 'not_comparable' for x in rows)
        if change in ('extra', 'period'):
            assert len(rows) == 6
        if change == 'kind':
            assert all('credit_kind_unsupported' in x.reasons for x in rows)
        if change == 'unproven':
            assert all('credit_identity_unproven' in x.reasons for x in rows)


@pytest.mark.parametrize('change', ['digest', 'pin_meter', 'key_run', 'key_content', 'key_version',
                                  'missing', 'hash', 'input', 'snapshot', 'duplicate_conflict',
                                  'same_role', 'historical_role'])
def test_e1b_source_pin_keys_and_duplicate_versions(e1b_case, change):
    view, supplied, sources = e1b_credit(e1b_case())
    supplied = replace(supplied, total=Decimal('0.15'))
    first = sources.lines[0]
    if change == 'digest':
        view = replace(view, lines=(replace(view.lines[0], import_cost=Decimal('0.11')), *view.lines[1:]))
    elif change == 'pin_meter':
        sources = replace(sources, pin=replace(sources.pin, meter_id='wrong'))
    elif change.startswith('key_'):
        key = json.loads(first.source_key)
        key[{'key_run': 4, 'key_content': 5, 'key_version': 3}[change]] = 2 if change == 'key_version' else 'wrong'
        sources = replace(sources, lines=(replace(first, source_key=r._canonical_json(key)), *sources.lines[1:]))
    elif change in ('missing', 'hash', 'input', 'snapshot'):
        changes = {'missing': {'item': None, 'issues': ('source_missing',)},
                   'hash': {'input_content_hash': 'wrong'},
                   'input': {'item': replace(first.item, meter_id='wrong')},
                   'snapshot': {'snapshot': replace(first.snapshot, contract_id='wrong')}}[change]
        sources = replace(sources, lines=(replace(first, **changes), *sources.lines[1:]))
    elif change == 'duplicate_conflict':
        cs = sources.credits[0]
        sources = replace(sources, credits=(replace(cs, closing_source=replace(cs.closing_source,
                                        input_content_hash='wrong')),))
    elif change == 'historical_role':
        # Same input, new state/content on generation 2; credit retains its generation-1 closing line.
        old = view.lines[-1]
        newer = replace(old, bonus_revenue=Decimal('0.01'))
        last = e1b_line_source(newer, sources.lines[-1].item, sources.lines[-1].snapshot, 2, 'run-2')
        view = replace(view, generation=2, lines=(*view.lines[:-1], newer),
                       net_cost=Decimal('0.14'), known_subtotal=Decimal('0.14'))
        run = replace(sources.runs[0], run_id='run-2', generation=2)
        sources = e1b_repin(view, replace(sources, lines=(*sources.lines[:-1], last),
                                         runs=(*sources.runs, run), pin=replace(sources.pin, current_generation=2)))
        supplied = replace(supplied, bonus_amounts=e1b_amounts('0.01'), total=Decimal('0.14'))
    report = e1b_compare((view, supplied, sources))
    rows = e1b_rows(report)
    if change in ('same_role', 'historical_role'):
        assert rows['import.gross'].book_value == Decimal('0.20')
        assert report.total.delta == 0
        for row in report.components:
            assert len(row.source_keys) == len(set(row.source_keys))
        assert all(x.status == 'matched' for x in report.components if x.key.startswith('credit:'))
    else:
        assert report.status == 'not_comparable'
        assert report.total.delta is None
        if change in ('digest', 'pin_meter'):
            assert all(x.status == 'not_comparable' for x in report.components)
        assert any(('source_missing' if change == 'missing' else 'source_mismatch') in x.reasons
                   for x in (*report.components, report.total))
    assert report.provenance == sources


@pytest.mark.parametrize('change', ['engine', 'rule', 'import_null', 'gross_null', 'energy_null', 'credit_null'])
def test_e1b_unknown_versions_and_nulls(e1b_case, change):
    view, supplied, sources = e1b_case()
    if change == 'engine':
        sources = replace(sources, runs=(replace(sources.runs[0], engine_version='future'),))
    elif change == 'rule':
        view, supplied, sources = e1b_case(terms={'rule_version': 'future'})
    elif change == 'import_null':
        supplied = replace(supplied, import_amounts=None)
    elif change == 'gross_null':
        supplied = replace(supplied, import_amounts=replace(supplied.import_amounts, gross=None))
    elif change == 'energy_null':
        supplied = replace(supplied, import_kwh=None)
    else:
        supplied = replace(supplied, credits=None)
    report = e1b_compare((view, supplied, sources))
    assert len(report.components) == 11 and report.status == 'not_comparable'
    if change in ('engine', 'rule'):
        assert e1b_rows(report)['import_kwh'].book_value == 2
        assert e1b_rows(report)['import_kwh'].status == 'matched'
        assert 'source_version_unsupported' in e1b_rows(report)['import.gross'].reasons
    else:
        assert any(c.delta is None for c in (*report.components, report.total))


@pytest.mark.parametrize('change', ['gap', 'cut', 'overlap', 'contract', 'hash_switch', 'meter',
                                  'period', 'currency', 'unit', 'sign', 'scope'])
def test_e1b_coverage_dst_and_contract_scope_invalid(e1b_case, change):
    view, supplied, sources = e1b_case()
    if change in ('gap', 'overlap'):
        view = replace(view, lines=view.lines[:1] if change == 'gap' else (*view.lines, view.lines[0]))
        sources = e1b_repin(view, sources)
    elif change == 'cut':
        sources = replace(sources, pin=replace(sources.pin, start_ts=sources.pin.start_ts + 1))
        supplied = replace(supplied, start_ts=supplied.start_ts + 1)
    elif change in ('contract', 'hash_switch'):
        first = sources.lines[0]
        snap = e1b_hash_snapshot(replace(first.snapshot, contract_id='other' if change == 'contract' else 'c',
                                       source_versions=(('synthetic', 'v2'),)))
        line = replace(view.lines[0], snapshot_hash=snap.snapshot_hash)
        source = e1b_line_source(line, first.item, snap)
        view = replace(view, lines=(line, *view.lines[1:]))
        sources = e1b_repin(view, replace(sources, lines=(source, *sources.lines[1:]),
                    runs=(replace(sources.runs[0], snapshot_hashes=(*sources.runs[0].snapshot_hashes,
                                                                   snap.snapshot_hash)),)))
    else:
        field, value = {'meter': ('meter_id', 'other'), 'period': ('end_ts', supplied.end_ts + 900),
                        'currency': ('currency', 'USD'), 'unit': ('energy_unit', 'Wh'),
                        'sign': ('sign_convention', 'net'), 'scope': ('total_scope', 'consumption')}[change]
        supplied = replace(supplied, **{field: value})
    report = e1b_compare((view, supplied, sources))
    if change == 'hash_switch':
        assert report.status == 'matched'
    else:
        assert report.status == 'not_comparable' and report.total.delta is None
        if change == 'contract':
            assert 'contract_scope_unsupported' in report.reasons


@pytest.mark.parametrize('day,hours', [('2027-03-28', 23), ('2027-10-31', 25)])
def test_e1b_coverage_dst_and_contract_scope_calendar(e1b_case, day, hours):
    from zoneinfo import ZoneInfo
    start = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Amsterdam')).timestamp())
    count = hours * 4
    view, supplied, sources = e1b_case(start=start, spots=('0.10',) * count)
    supplied = replace(supplied, import_kwh=Decimal(str(count)),
                       import_amounts=e1b_amounts('9.20' if hours == 23 else '10.00'),
                       total=Decimal('9.20' if hours == 23 else '10.00'))
    report = e1b_compare((view, supplied, sources))
    assert report.status == 'matched' and report.total.delta == 0
    if hours == 25:
        instants = [x.start_ts for x in sources.lines]
        for instant in ('2027-10-31T00:00:00+00:00', '2027-10-31T01:00:00+00:00'):
            assert int(datetime.fromisoformat(instant).timestamp()) in instants


@pytest.mark.parametrize('freshness', ['current', 'outdated', 'unknown'])
def test_e1b_freshness_and_report_precedence(e1b_case, freshness):
    view, supplied, sources = e1b_case()
    view = replace(view, freshness=freshness)
    sources = e1b_repin(view, sources)
    supplied = replace(supplied, import_amounts=e1b_amounts('0.21'), total=Decimal('0.21'))
    report = e1b_compare((view, supplied, sources))
    assert report.freshness == freshness
    if freshness == 'current':
        assert report.status == 'different'
    elif freshness == 'outdated':
        assert e1b_rows(report)['import.gross'].delta == Decimal('0.01')
        assert report.status == 'not_comparable' and 'book_outdated' in report.reasons
    else:
        assert report.status == 'not_comparable' and report.total.delta is None


@pytest.mark.parametrize('precision', [6, 28, 50])
def test_e1b_context_independent_stored_sums(e1b_case, precision):
    case = e1b_case(gross=('1000000000000000000000000.005', '0.005'))
    view, supplied, sources = case
    with localcontext() as ctx:
        ctx.prec = precision
        report = e1b_compare(case)
    row = e1b_rows(report)['import.gross']
    assert row.book_value == Decimal('1000000000000000000000000.010')
    assert row.comparable_book_value == Decimal('1000000000000000000000000.01')
    assert report.total.book_value == view.net_cost
    # The book credit contract is context36, not a new exact high-precision credit sum.
    case = e1b_credit(e1b_case(), ex='1.00000000000000000000000000000000000',
                      vat='0.000000000000000000000000000000000005', gross='1',
                      supplier_amounts=e1b_amounts('1'))
    # Supplier resource limits are independent of the lossless book operands.
    with localcontext() as ctx:
        ctx.prec = precision
        report = e1b_compare(case)
    assert all('source_mismatch' not in x.reasons for x in report.components if x.key.startswith('credit:'))


def test_e1b_noop_global_revision_above_run_revision_keeps_comparability(repo, terms, inputs, calculate):
    """Een schone no-op heft de globale revisie zonder een nieuwe run te publiceren."""
    snapshots = [terms(rule_version="tariffs-v1")]
    publish(repo, inputs(snapshots), snapshots, calculate)
    # Een prijs buiten de geboekte dekking verhoogt alleen de globale invoerrevisie.
    assert repo.register_prices(METER, [SpotInterval(START_TS + 3600, START_TS + 7200, Decimal("0.30"),
                                                     "synthetic-hour-price", "measured")], 1) == 2
    batch = repo.read_inputs(METER, START_TS, START_TS + 1800)
    # Het onveranderde resultaat is een no-op: geen nieuwe generatie, geen nieuwe run.
    assert repo.commit(METER, calculate(batch), batch, 1, "synthetic-noop") == 1
    basis = repo.read_reconciliation(METER, START_TS, START_TS + 1800)
    run = basis.sources.runs[0]
    assert (basis.view.input_revision, run.input_revision) == (2, 1)
    assert basis.view.input_revision > run.input_revision
    assert basis.view.freshness == "current"
    assert basis.sources.pin.view_input_revision == basis.view.input_revision
    supplied = supplier(meter_id=METER, contract_id="synthetic-2027", start_ts=START_TS,
                        end_ts=START_TS + 1800, import_kwh=Decimal("2"), export_kwh=Decimal("0"),
                        import_amounts=e1b_amounts("0.20"), base_export_amounts=e1b_amounts(),
                        bonus_amounts=e1b_amounts(), credits=(), total=Decimal("0.20"),
                        rounding=e1b_rounding())
    report = r.reconcile(basis.view, supplied, sources=basis.sources)
    assert "source_mismatch" not in report.reasons
    assert all("source_mismatch" not in row.reasons for row in (*report.components, report.total))
    assert all(row.comparable_book_value is not None and row.delta == 0 and row.status == "matched"
               for row in (*report.components, report.total))
    assert report.status == "matched"


@pytest.mark.parametrize("change", ["run_revision_higher", "conflicting_run"])
def test_e1b_run_revision_above_view_revision_stays_source_mismatch(e1b_case, change):
    """Een run boven de viewrevisie of een tweede run voor dezelfde generatie blijft mismatch."""
    view, supplied, sources = e1b_case()
    if change == "run_revision_higher":
        sources = replace(sources, runs=(replace(sources.runs[0], input_revision=view.input_revision + 1),))
    else:
        sources = replace(sources, runs=(*sources.runs, replace(sources.runs[0], run_id="run-2")))
    report = e1b_compare((view, supplied, sources))
    assert "source_mismatch" in report.reasons
    assert report.status == "not_comparable" and report.total.delta is None
    assert all("source_mismatch" in row.reasons for row in (*report.components, report.total))


def supplier(**changes):
    return r.SupplierPeriod(**{
        "schema_version": "supplier-period-v1", "meter_id": "m", "contract_id": "c",
        "timezone": "Europe/Amsterdam", "start_ts": 0, "end_ts": 900,
        "currency": "EUR", "energy_unit": "kWh",
        "sign_convention": "cost_positive_revenue_positive", "total_scope": "book_close_events_v1",
        "import_kwh": None, "export_kwh": None, "import_amounts": None,
        "base_export_amounts": None, "bonus_amounts": None, "credits": None,
        "total": None, "rounding": None, **changes,
    })


def credit(**changes):
    return r.SupplierCredit(**{
        "kind": "spot_zero", "period_start_ts": 0, "period_end_ts": 900,
        "amounts": r.SupplierAmounts(None, None, None), **changes,
    })


def rounding(**changes):
    return r.SupplierRounding(**{
        "method": "declared", "money_quantum": None, "money_mode": None,
        "aggregation": None, "total_rule": None, "energy_quantum": None,
        "energy_mode": None, "evidence_ref": None, **changes,
    })


def examples():
    state = SettlementState(0, None, None)
    line = SettlementLine(
        start_ts=0, end_ts=900, meter_id="m", input_revision=1, input_hash="i",
        snapshot_hash=None, import_cost=Decimal("1.123456789012345678901234567890123456789"),
        base_export_revenue=None, bonus_revenue=None, bonus_kwh=None,
        before_state=state, after_state=replace(state, asof_ts=900), quality="incomplete", issues=(),
    )
    view = SettlementView(
        meter_id="m", generation=1, input_revision=1, current_input_revision=1,
        freshness="current", lines=(line,), adjustments=(), known_subtotal=line.import_cost,
        net_cost=None, coverage_start_ts=0, coverage_end_ts=900, complete=False,
        captured_ts=900, source_watermark_ts=None, dirty_from_ts=None,
    )
    pin = r.BookSourcePin("m", 1, 1, 1, 1, None, 0, 900, "digest", 901)
    run = r.RunSource("r", 1, 1, "v1", "i", "result", ["snapshot"])
    source = r.LineSource(
        '["settlement-line-v1","m",0,1,"r","h"]', 0, 900, 1, "i", None,
        1, "r", "h", None, None, None, ["source_missing"],
    )
    cs = r.CreditSource(
        '["settlement-credit-v1","m","spot_zero",0,900,1,"r","h"]',
        "spot_zero", 0, 900, 1, "r", "h", None, None, line, source, [],
    )
    sources = r.ReconciliationSources(pin, [run], [source], [cs], [])
    component = r.ComponentComparison(
        "net_cost", "EUR", line.import_cost, None, None, None,
        "not_comparable", "incomplete", ["supplier_missing"], [source.source_key],
    )
    period = supplier(credits=[credit()], rounding=rounding())
    report = r.ReconciliationReport(
        "reconciliation-report-v1", pin, period, "supplier-digest",
        "registered_book_and_caller_supplied_supplier", "current", [component],
        component, "not_comparable", [], sources,
    )
    return dict(zip(DTO_NAMES, (
        r.SupplierAmounts(None, None, None), credit(), rounding(), period, pin, run,
        source, cs, sources, component, report, r.ReconciliationBasis(view, sources),
    ), strict=True))


@pytest.mark.parametrize("name", DTO_NAMES)
def test_e1a_required_fields_and_explicit_nulls(name):
    obj = examples()[name]
    values = {f.name: getattr(obj, f.name) for f in fields(obj)}
    for f in fields(obj):
        assert f.default is MISSING and f.default_factory is MISSING
        with pytest.raises(TypeError):
            type(obj)(**{key: value for key, value in values.items() if key != f.name})
        if type(None) in get_args(get_type_hints(type(obj))[f.name]):
            assert getattr(replace(obj, **{f.name: None}), f.name) is None


def assert_immutable(value):
    assert not isinstance(value, (dict, list, set))
    if is_dataclass(value):
        assert not hasattr(value, "__dict__")
        for f in fields(value):
            with pytest.raises((AttributeError, TypeError)):
                setattr(value, f.name, getattr(value, f.name))
            assert_immutable(getattr(value, f.name))
    elif isinstance(value, tuple):
        for item in value:
            assert_immutable(item)


@pytest.mark.parametrize("name", DTO_NAMES)
def test_e1a_recursive_immutability_and_detachment(name):
    obj = examples()[name]
    values = {f.name: getattr(obj, f.name) for f in fields(obj)}
    lists = {key: list(value) for key, value in values.items() if isinstance(value, tuple)}
    copy = type(obj)(**(values | lists))
    for value in lists.values():
        value.clear()
        value.append({"mutable": []})
    assert copy == obj
    assert_immutable(copy)
    for key in lists:
        with pytest.raises((TypeError, ValueError)):
            replace(obj, **{key: [{"mutable": []}]})


@pytest.mark.parametrize("name", DTO_NAMES)
def test_e1a_strict_types_and_decimal_limits_declared_fields(name):
    obj = examples()[name]
    for f in fields(obj):
        with pytest.raises((TypeError, ValueError)):
            replace(obj, **{f.name: {"wrong": "type"}})


@pytest.mark.parametrize("value", [True, 1, 0.1, "NaN", "Infinity", "-Infinity", Decimal("NaN"),
                                   Decimal("Infinity"), "1e2", "1E-2", " 1", "1_000"])
def test_e1a_strict_types_and_decimal_limits_invalid_decimal(value):
    for build in (lambda: r.SupplierAmounts(value, None, None),
                  lambda: supplier(import_kwh=value), lambda: supplier(total=value),
                  lambda: rounding(money_quantum=value), lambda: rounding(energy_quantum=value)):
        with pytest.raises((TypeError, ValueError)):
            build()


@pytest.mark.parametrize("field", ["import_kwh", "export_kwh"])
@pytest.mark.parametrize("value", ["-0.1", "0.0000000001"])
def test_e1a_strict_types_and_decimal_limits_energy(field, value):
    with pytest.raises(ValueError):
        supplier(**{field: value})
    assert getattr(supplier(**{field: "0.123456789"}), field) == Decimal("0.123456789")


@pytest.mark.parametrize("target,field,good,bad", [
    ("amounts", "gross", "0" * 63 + "1", "0" * 64 + "1"),
    ("amounts", "vat", "9" * 36, "9" * 37),
    ("amounts", "ex_vat", "-0.123456789012", "-0.1234567890123"),
    ("supplier", "total", "0.123456789012", "0.1234567890123"),
    ("rounding", "money_quantum", "0.123456789012", "0.1234567890123"),
    ("rounding", "energy_quantum", "0.123456789012", "0.1234567890123"),
])
def test_e1a_strict_types_and_decimal_limits_boundaries(target, field, good, bad):
    obj = {"amounts": r.SupplierAmounts(None, None, None),
           "supplier": supplier(), "rounding": rounding()}[target]
    assert getattr(replace(obj, **{field: good}), field) == Decimal(good)
    with pytest.raises(ValueError):
        replace(obj, **{field: bad})


@pytest.mark.parametrize("field", ["meter_id", "contract_id", "currency", "energy_unit",
                                  "sign_convention", "total_scope"])
def test_e1a_supplier_scope_shape_identifiers(field):
    assert getattr(supplier(**{field: "x" * 128}), field) == "x" * 128
    for value in ("x" * 129, "x\n", "x\x7f", "x\x85"):
        with pytest.raises(ValueError):
            supplier(**{field: value})


def test_e1a_supplier_scope_shape_labels_and_unsupported():
    assert rounding(evidence_ref="x" * 256).evidence_ref == "x" * 256
    for value in ("x" * 257, "x\n"):
        with pytest.raises(ValueError):
            rounding(evidence_ref=value)
    assert credit(kind="other").kind == "other"
    assert rounding(method="other").method == "other"
    assert supplier(currency="USD", energy_unit="MWh").energy_unit == "MWh"
    assert supplier(credits=None).credits is None
    assert supplier(credits=[]).credits == ()
    assert r.SupplierAmounts("-1", "-2", "-3").gross == Decimal("-3")
    for obj, field in ((credit(), "kind"), (rounding(), "method"),
                       (rounding(), "money_mode"), (rounding(), "energy_mode"),
                       (rounding(), "aggregation"), (rounding(), "total_rule")):
        assert getattr(replace(obj, **{field: "x" * 128}), field) == "x" * 128
        with pytest.raises(ValueError):
            replace(obj, **{field: "x" * 129})


@pytest.mark.parametrize("value", [True, 0.0, "0", -62135596801, 253402300800])
def test_e1a_supplier_scope_shape_epochs(value):
    for build in (lambda: supplier(start_ts=value), lambda: supplier(end_ts=value),
                  lambda: credit(period_start_ts=value), lambda: credit(period_end_ts=value)):
        with pytest.raises((TypeError, ValueError)):
            build()


def test_e1a_supplier_scope_shape_ranges_and_schema():
    assert supplier(start_ts=-62135596800, end_ts=-62135596799).start_ts == -62135596800
    assert supplier(start_ts=253402300798, end_ts=253402300799).end_ts == 253402300799
    assert supplier(end_ts=370 * 86400).end_ts == 370 * 86400
    for changes in ({"end_ts": 370 * 86400 + 1}, {"end_ts": 0}, {"start_ts": 901},
                    {"schema_version": "supplier-period-v2"}, {"timezone": "Not/AZone"},
                    {"timezone": ""}):
        with pytest.raises(ValueError):
            supplier(**changes)
    for end in (0, -1):
        with pytest.raises(ValueError):
            credit(period_end_ts=end)
    report = examples()["ReconciliationReport"]
    with pytest.raises(ValueError):
        replace(report, schema_version="reconciliation-report-v2")


def test_e1a_supplier_scope_shape_credit_matrix():
    credits = [credit(period_start_ts=i, period_end_ts=i + 1) for i in range(65)]
    assert len(supplier(credits=credits[:64]).credits) == 64
    for items in (credits, [credit(), credit()],
                  [credit(), credit(period_start_ts=899, period_end_ts=901)]):
        with pytest.raises(ValueError):
            supplier(credits=items)
    assert len(supplier(credits=[credit(), credit(kind="other")]).credits) == 2
    assert len(supplier(credits=list(reversed(credits[:64]))).credits) == 64


def test_e1a_canonical_digest_and_source_key_shape():
    objects = examples()
    view = objects["ReconciliationBasis"].view
    assert r._canonical_json(view) == _json(view)
    assert r._digest(view) == sha256(_json(view).encode("utf-8")).hexdigest()
    changed = replace(view, lines=(replace(view.lines[0], import_cost=Decimal("2")),))
    assert r._digest(view) != r._digest(changed)
    assert r._canonical_json(["settlement-line-v1", "m", 0, 1, "r", "h"]) == (
        '["settlement-line-v1","m",0,1,"r","h"]'
    )
    assert r._canonical_json(["settlement-credit-v1", "m", "spot_zero", 0, 900, 1, "r", "h"]) == (
        objects["CreditSource"].source_key
    )
    assert r._canonical_json(["mé\"", 0]) == '["m\\u00e9\\\"",0]'
    assert r._canonical_json({"z": Decimal("1.00"), "a": Decimal("-0.00")}) == '{"a":"0","z":"1"}'
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            r._canonical_json(value)
    assert objects["ComponentComparison"].book_value == view.lines[0].import_cost
    assert json.loads(r._canonical_json(view))["lines"][0]["import_cost"] == (
        "1.123456789012345678901234567890123456789"
    )
    assert r._digest(supplier(total="1.00")) == r._digest(supplier(total=Decimal("1")))


def test_e1a_module_is_pure():
    tree = ast.parse(inspect.getsource(r))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    forbidden = {"repository", "config", "os", "pathlib", "socket", "requests", "time", "io"}
    assert all(not forbidden.intersection(module.split(".")) for module in imports)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not {"open", "datetime.now", "datetime.today", "date.today", "__import__"} & calls
