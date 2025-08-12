#!/usr/bin/env python3
"""
Test van de Day Ahead Optimalisatie functionaliteit
"""

import sys
import os
import datetime
import logging
import pandas as pd

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
prog_dir = os.path.join(parent_dir, 'prog')
if prog_dir not in sys.path:
    sys.path.insert(0, prog_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_get_grid_data_sqlite():
    """Test grid data retrieval from SQLite database"""
    try:
        from da_report import Report

        # Test with SQLite database
        report = Report("../data/options_sqlite.json")

        # Test data for a specific day
        day = datetime.datetime(2024, 7, 9)
        vanaf = day
        tot = day + datetime.timedelta(days=1)  # datetime.datetime(2024, 7, 10)

        df_ha = report.get_grid_data(
            periode="", _vanaf=vanaf, _tot=tot, _interval="uur", _source="ha"
        )
        df_ha = report.calc_grid_columns(df_ha, "uur", "tabel")
        logging.info(
            f"Eigen meterstanden op {day.strftime('%Y-%m-%d')}:\n{df_ha.to_string(index=False)}"
        )
        df_da = report.get_grid_data(
            periode="", _vanaf=vanaf, _tot=tot, _interval="uur", _source="da"
        )
        df_da = report.calc_grid_columns(df_da, "uur", "tabel")
        logging.info(
            f"Verbruiken gecorrigeerd door Tibber op {day.strftime('%Y-%m-%d')}:\n"
            f"{df_da.to_string(index=False)}"
        )
        # logging.debug(f"Data comparison: {df_ha.equals(df_da)}")

        return True

    except Exception as e:
        logging.error(f"Test failed: {e}")
        return False


def start_logging():
    """Initialize logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(
        f"Testen Day Ahead Optimalisatie gestart: "
        f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )


def test_da_calc():
    """Test the main DAO calculation engine"""
    start_logging()
    try:
        # da_calc = dao.prog.day_ahead.DaCalc(file_name="../data/options_mysql.json")
        da_calc = dao.prog.day_ahead.DaCalc(file_name="../data/options_hetzerha.json")
        da_calc.calc_optimum(
            _start_dt=datetime.datetime(year=2025, month=1, day=26, hour=19, minute=0),
            _start_soc=30.0,
        )
        """
        da_calc.calc_optimum(
            _start_dt=datetime.datetime(year=2024, month=9, day=21, hour=14, minute=0),
            _start_soc=35.0,
        )
        # da_calc.calc_optimum(_start_soc=67.2)
        """
        return True

    except Exception as e:
        logging.error(f"DA calculation test failed: {e}")
        return False


def get_grid_data(
    engine: str,
    source: str,
    vanaf: datetime.datetime,
    tot: datetime.datetime = None,
    interval: str = "uur",
) -> tuple:
    """Get grid data from different database engines and sources"""
    try:
        file_name = "../data/options_" + engine + ".json"
        report = dao.prog.da_report.Report(file_name)
        if tot is None:
            tot = vanaf + datetime.timedelta(days=1)
        df = report.get_grid_data(
            periode="", _vanaf=vanaf, _tot=tot, _interval=interval, _source=source
        )
        df = report.calc_grid_columns(df, interval, "tabel")
        row = df.iloc[-1]
        netto_consumption = row.Verbruik[0] - row.Productie[0]
        netto_kosten = row.Kosten[0] - row.Opbrengst[0]
        return df, netto_consumption, netto_kosten

    except Exception as e:
        logging.error(f"Failed to get grid data for {engine}/{source}: {e}")
        return None, 0, 0


def test_grid_reporting():
    """Test grid reporting across different database engines"""
    engines = ["mysql", "sqlite", "postgresql"]
    sources = ["da", "ha"]
    result = [
        pd.DataFrame(columns=["engine", "netto_consumption", "netto_cost"]),
        pd.DataFrame(columns=["engine", "netto_consumption", "netto_cost"]),
    ]

    for engine in engines:
        for s in range(len(sources)):
            try:
                vanaf = datetime.datetime(2024, 8, 13)
                df, netto_consumption, netto_cost = get_grid_data(engine, sources[s], vanaf)

                if df is not None:
                    logging.info(
                        f"Result from source:{sources[s]} engine:{engine} :\n{df.to_string(index=False)}"
                    )
                    result[s].loc[result[s].shape[0]] = [engine, netto_consumption, netto_cost]
                else:
                    logging.warning(f"No data returned for {engine}/{sources[s]}")

            except Exception as e:
                logging.error(f"Error testing {engine}/{sources[s]}: {e}")

    logging.info(f"Result from DA:\n{result[0].to_string(index=False)}")
    logging.info(f"Result from HA:\n{result[1].to_string(index=False)}")

    return result


def test_report_start_periode():
    """Test report generation for different time periods"""
    try:
        file_name = "../data/options_mysql.json"
        report = dao.prog.da_report.Report(
            file_name, _now=datetime.datetime(year=2022, month=7, day=1)
        )
        df = report.get_grid_data(periode="vorige maand")
        df = report.calc_grid_columns(df, "dag", "tabel")
        logging.info(f"Result test start periode:\n{df.to_string(index=False)}")
        return True

    except Exception as e:
        logging.error(f"Report period test failed: {e}")
        return False


def test_main():
    """Main test function"""
    logging.info("Starting DAO tests...")

    tests = [
        ("Grid Data SQLite", test_get_grid_data_sqlite),
        ("DA Calculation", test_da_calc),
        ("Grid Reporting", test_grid_reporting),
        ("Report Periods", test_report_start_periode)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            logging.info(f"{status}: {test_name}")
        except Exception as e:
            logging.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0

    logging.info(f"\nTest Results: {passed}/{total} passed ({success_rate:.1f}%)")

    if passed == total:
        logging.info("🎉 All tests passed!")
    else:
        logging.warning("⚠️  Some tests failed")

    return passed == total


if __name__ == "__main__":
    success = test_main()
    sys.exit(0 if success else 1)
