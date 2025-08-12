#!/usr/bin/env python3
"""
Battery Controller voor Day Ahead Optimizer
Bepaalt en stelt batterij setpoints in op basis van optimalisatie resultaten
"""
import logging
import datetime as dt
from typing import Dict, Any, Optional, Tuple
import requests
import json

class BatteryController:
    """Beheert batterij controle en setpoint instellingen"""

    def __init__(self, config, ha_url: str = None, ha_token: str = None):
        self.config = config
        self.ha_url = ha_url or config.get_homeassistant_url()
        self.ha_token = ha_token or config.get_homeassistant_token()
        self.battery_entity_id = config.get_battery_entity_id()
        self.setpoint_entity_id = config.get_battery_setpoint_entity_id()
        self.control_enabled = config.get_battery_control_enabled()

        # Batterij limieten
        self.min_soc = config.get_battery_min_soc()
        self.max_soc = config.get_battery_max_soc()
        self.max_charge_rate = config.get_battery_max_charge_rate()
        self.max_discharge_rate = config.get_battery_max_discharge_rate()

        # Batterij efficiëntie curves
        self.charge_efficiency_curve = self._parse_efficiency_curve(
            config.get_battery_charge_efficiency_curve(), "charge"
        )
        self.discharge_efficiency_curve = self._parse_efficiency_curve(
            config.get_battery_discharge_efficiency_curve(), "discharge"
        )
        self.battery_capacity = config.get_battery_capacity()

                logging.info(f"Battery controller geïnitialiseerd: control_enabled={self.control_enabled}")
        if self.control_enabled:
            logging.info(f"Battery entity: {self.battery_entity_id}")
            logging.info(f"Setpoint entity: {self.setpoint_entity_id}")
            logging.info(f"Battery capacity: {self.battery_capacity} kWh")
            logging.info(f"Charge efficiency curve: {len(self.charge_efficiency_curve)} points")
            logging.info(f"Discharge efficiency curve: {len(self.discharge_efficiency_curve)} points")

    def _parse_efficiency_curve(self, curve_string: str, curve_type: str) -> Dict[float, float]:
        """
        Parse efficiëntie curve string naar dictionary

        Args:
            curve_string: String in formaat "0:85,10:90,20:93,50:95,80:92,90:88,95:82"
            curve_type: Type curve ("charge" of "discharge")

        Returns:
            Dictionary met SoC als key en efficiëntie als value
        """
        try:
            if not curve_string:
                # Standaard efficiëntie curve als geen configuratie
                if curve_type == "charge":
                    return {0: 0.85, 10: 0.90, 20: 0.93, 50: 0.95, 80: 0.92, 90: 0.88, 95: 0.82}
                else:  # discharge
                    return {0: 0.75, 10: 0.88, 20: 0.92, 50: 0.95, 80: 0.94, 90: 0.91, 100: 0.85}

            curve_dict = {}
            points = curve_string.split(',')

            for point in points:
                if ':' in point:
                    soc_str, efficiency_str = point.strip().split(':')
                    try:
                        soc = float(soc_str)
                        efficiency = float(efficiency_str) / 100.0  # Convert percentage to decimal
                        curve_dict[soc] = efficiency
                    except ValueError:
                        logging.warning(f"Ongeldig punt in {curve_type} efficiëntie curve: {point}")
                        continue

            # Sorteer op SoC
            sorted_curve = dict(sorted(curve_dict.items()))

            if not sorted_curve:
                logging.warning(f"Geen geldige punten gevonden in {curve_type} efficiëntie curve, gebruik standaard")
                return self._parse_efficiency_curve("", curve_type)

            logging.info(f"{curve_type.capitalize()} efficiëntie curve geparsed: {len(sorted_curve)} punten")
            return sorted_curve

        except Exception as e:
            logging.error(f"Fout bij parsen {curve_type} efficiëntie curve: {e}")
            return self._parse_efficiency_curve("", curve_type)

    def _get_efficiency_at_soc(self, soc: float, curve_type: str) -> float:
        """
        Haalt efficiëntie op voor gegeven SoC uit de curve

        Args:
            soc: Batterij SoC (%)
            curve_type: Type curve ("charge" of "discharge")

        Returns:
            Efficiëntie als decimal (0.0 - 1.0)
        """
        try:
            if curve_type == "charge":
                curve = self.charge_efficiency_curve
            else:
                curve = self.discharge_efficiency_curve

            if not curve:
                return 0.95  # Standaard efficiëntie

            # Zoek het dichtstbijzijnde punt of interpolate
            soc_values = list(curve.keys())

            if soc <= soc_values[0]:
                return curve[soc_values[0]]
            elif soc >= soc_values[-1]:
                return curve[soc_values[-1]]

            # Interpolate tussen twee punten
            for i in range(len(soc_values) - 1):
                if soc_values[i] <= soc <= soc_values[i + 1]:
                    soc1, soc2 = soc_values[i], soc_values[i + 1]
                    eff1, eff2 = curve[soc1], curve[soc2]

                    # Lineaire interpolatie
                    ratio = (soc - soc1) / (soc2 - soc1)
                    interpolated_efficiency = eff1 + ratio * (eff2 - eff1)

                    return round(interpolated_efficiency, 3)

            return 0.95  # Fallback

        except Exception as e:
            logging.error(f"Fout bij ophalen efficiëntie voor SoC {soc}: {e}")
            return 0.95

    def _calculate_effective_power(self, requested_power: float, soc: float, action: str) -> float:
        """
        Berekent effectief vermogen rekening houdend met efficiëntie

        Args:
            requested_power: Gewenst vermogen (kW)
            soc: Huidige batterij SoC (%)
            action: Actie ("charge" of "discharge")

        Returns:
            Effectief vermogen (kW)
        """
        try:
            if requested_power == 0:
                return 0.0

            efficiency = self._get_efficiency_at_soc(soc, action)

            if action == "charge":
                # Voor laden: effectief vermogen = gewenst vermogen / efficiëntie
                # (we moeten meer vermogen leveren om het gewenste effect te krijgen)
                effective_power = requested_power / efficiency
            else:  # discharge
                # Voor ontladen: effectief vermogen = gewenst vermogen * efficiëntie
                # (we krijgen minder vermogen dan we leveren)
                effective_power = requested_power * efficiency

            return round(effective_power, 3)

        except Exception as e:
            logging.error(f"Fout bij berekenen effectief vermogen: {e}")
            return requested_power

        def calculate_battery_setpoint(self,
                                 current_soc: float,
                                 target_soc: float,
                                 time_to_target: float,
                                 price_context: Dict[str, Any] = None) -> Tuple[float, str]:
        """
        Berekent de optimale batterij setpoint rekening houdend met efficiëntie curves

        Args:
            current_soc: Huidige batterij SoC (%)
            target_soc: Gewenste batterij SoC (%)
            time_to_target: Tijd tot doel in uren
            price_context: Prijs context voor optimalisatie

        Returns:
            Tuple van (setpoint_kw, reason)
        """
        if not self.control_enabled:
            return 0.0, "Batterij controle uitgeschakeld"

        if not self.setpoint_entity_id:
            return 0.0, "Geen setpoint entity geconfigureerd"

        try:
            # Bereken SoC verschil
            soc_difference = target_soc - current_soc

            if abs(soc_difference) < 1.0:
                return 0.0, "SoC al op gewenste niveau"

            # Bepaal actie type
            if soc_difference > 0:
                action = "charge"
                reason = f"Laden naar {target_soc}% SoC"
            else:
                action = "discharge"
                reason = f"Ontladen naar {target_soc}% SoC"

            # Bereken benodigde vermogen rekening houdend met efficiëntie
            if time_to_target > 0:
                # Bereken gewenst effectief vermogen
                required_energy = abs(soc_difference) * self.battery_capacity / 100.0  # kWh
                required_power = required_energy / time_to_target  # kW

                # Pas efficiëntie toe om effectief vermogen te berekenen
                effective_power = self._calculate_effective_power(required_power, current_soc, action)

                # Pas limieten toe
                if action == "charge":
                    max_power = min(effective_power, self.max_charge_rate)
                else:  # discharge
                    max_power = max(-effective_power, -self.max_discharge_rate)
            else:
                max_power = 0.0

            # Valideer SoC grenzen met efficiëntie
            if time_to_target > 0:
                # Bereken verwachte SoC verandering met efficiëntie
                if action == "charge":
                    # Voor laden: effectief vermogen * tijd * efficiëntie
                    efficiency = self._get_efficiency_at_soc(current_soc, "charge")
                    soc_change = (max_power * time_to_target * efficiency * 100.0) / self.battery_capacity
                else:
                    # Voor ontladen: effectief vermogen * tijd / efficiëntie
                    efficiency = self._get_efficiency_at_soc(current_soc, "discharge")
                    soc_change = -(max_power * time_to_target * 100.0) / (efficiency * self.battery_capacity)

                # Controleer SoC grenzen
                expected_soc = current_soc + soc_change

                if expected_soc > self.max_soc:
                    # Bereken maximaal toegestaan vermogen
                    max_soc_change = self.max_soc - current_soc
                    if action == "charge":
                        max_power = (max_soc_change * self.battery_capacity * 100.0) / (time_to_target * efficiency)
                    else:
                        max_power = -(max_soc_change * self.battery_capacity * 100.0) / (time_to_target * efficiency)
                    reason += f" (beperkt tot {self.max_soc}% SoC)"

                if expected_soc < self.min_soc:
                    # Bereken maximaal toegestaan vermogen
                    max_soc_change = current_soc - self.min_soc
                    if action == "charge":
                        max_power = (max_soc_change * self.battery_capacity * 100.0) / (time_to_target * efficiency)
                    else:
                        max_power = -(max_soc_change * self.battery_capacity * 100.0) / (time_to_target * efficiency)
                    reason += f" (beperkt tot {self.min_soc}% SoC)"

            # Rond af naar 2 decimalen
            setpoint = round(max_power, 2)

            # Log efficiëntie informatie
            efficiency = self._get_efficiency_at_soc(current_soc, action)
            logging.info(f"Battery setpoint berekend: {setpoint} kW ({reason}) - Efficiëntie bij {current_soc}% SoC: {efficiency:.1%}")

            return setpoint, reason

        except Exception as e:
            logging.error(f"Fout bij berekenen batterij setpoint: {e}")
            return 0.0, f"Fout: {str(e)}"

    def set_battery_setpoint(self, setpoint_kw: float, reason: str = "") -> bool:
        """
        Stelt de batterij setpoint in via Home Assistant

        Args:
            setpoint_kw: Setpoint in kW (positief = laden, negatief = ontladen)
            reason: Reden voor de setpoint wijziging

        Returns:
            True als succesvol, False anders
        """
        if not self.control_enabled:
            logging.warning("Batterij controle is uitgeschakeld")
            return False

        if not self.setpoint_entity_id:
            logging.error("Geen setpoint entity geconfigureerd")
            return False

        if not self.ha_url or not self.ha_token:
            logging.error("Home Assistant URL of token niet geconfigureerd")
            return False

        try:
            # Bereid de API call voor
            headers = {
                'Authorization': f'Bearer {self.ha_token}',
                'Content-Type': 'application/json'
            }

            # Converteer kW naar W (Home Assistant gebruikt meestal W)
            setpoint_w = int(setpoint_kw * 1000)

            # Maak de service call
            service_data = {
                'entity_id': self.setpoint_entity_id,
                'value': setpoint_w
            }

            # Roep de service aan
            url = f"{self.ha_url}/api/services/input_number/set_value"
            response = requests.post(url, headers=headers, json=service_data, timeout=10)

            if response.status_code == 200:
                logging.info(f"Battery setpoint ingesteld: {setpoint_kw} kW ({reason})")

                # Log de wijziging
                self._log_setpoint_change(setpoint_kw, reason)
                return True
            else:
                logging.error(f"Fout bij instellen setpoint: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logging.error(f"Fout bij instellen batterij setpoint: {e}")
            return False

    def get_battery_status(self) -> Dict[str, Any]:
        """
        Haalt de huidige batterij status op via Home Assistant

        Returns:
            Dictionary met batterij status informatie
        """
        if not self.control_enabled or not self.battery_entity_id:
            return {"error": "Batterij controle niet beschikbaar"}

        try:
            headers = {
                'Authorization': f'Bearer {self.ha_token}',
                'Content-Type': 'application/json'
            }

            url = f"{self.ha_url}/api/states/{self.battery_entity_id}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    "entity_id": self.battery_entity_id,
                    "state": data.get('state'),
                    "attributes": data.get('attributes', {}),
                    "last_updated": data.get('last_updated'),
                    "status": "online"
                }
            else:
                return {"error": f"HTTP {response.status_code}", "status": "offline"}

        except Exception as e:
            logging.error(f"Fout bij ophalen batterij status: {e}")
            return {"error": str(e), "status": "error"}

    def optimize_battery_schedule(self,
                                price_forecast: Dict[str, float],
                                consumption_forecast: Dict[str, float],
                                production_forecast: Dict[str, float],
                                current_soc: float) -> Dict[str, Any]:
        """
        Optimaliseert het batterij laad/ontlaad schema

        Args:
            price_forecast: Uurlijkse prijs voorspelling
            consumption_forecast: Uurlijkse consumptie voorspelling
            production_forecast: Uurlijkse productie voorspelling
            current_soc: Huidige batterij SoC

        Returns:
            Dictionary met optimalisatie resultaten
        """
        if not self.control_enabled:
            return {"error": "Batterij controle uitgeschakeld"}

        try:
            schedule = {}
            total_savings = 0.0

            # Eenvoudige optimalisatie strategie:
            # 1. Laad bij lage prijzen
            # 2. Ontlaad bij hoge prijzen
            # 3. Respecteer SoC grenzen

            for hour, price in price_forecast.items():
                # Bepaal of dit een goed moment is om te laden/ontladen
                if price < 0.20:  # Lage prijs - laden
                    target_soc = min(current_soc + 20, self.max_soc)
                    setpoint, reason = self.calculate_battery_setpoint(
                        current_soc, target_soc, 1.0, {"price": price}
                    )
                                    if setpoint > 0:
                    # Bereken effectieve SoC verandering met efficiëntie
                    efficiency = self._get_efficiency_at_soc(current_soc, "charge")
                    effective_soc_change = (setpoint * 1.0 * efficiency * 100.0) / self.battery_capacity

                    schedule[hour] = {
                        "action": "charge",
                        "setpoint": setpoint,
                        "reason": reason,
                        "price": price,
                        "efficiency": efficiency,
                        "effective_soc_change": effective_soc_change
                    }
                    current_soc += effective_soc_change

                elif price > 0.35:  # Hoge prijs - ontladen
                    target_soc = max(current_soc - 20, self.min_soc)
                    setpoint, reason = self.calculate_battery_setpoint(
                        current_soc, target_soc, 1.0, {"price": price}
                    )
                                    if setpoint < 0:
                    # Bereken effectieve SoC verandering met efficiëntie
                    efficiency = self._get_efficiency_at_soc(current_soc, "discharge")
                    effective_soc_change = -(setpoint * 1.0 * 100.0) / (efficiency * self.battery_capacity)

                    schedule[hour] = {
                        "action": "discharge",
                        "setpoint": setpoint,
                        "reason": reason,
                        "price": price,
                        "efficiency": efficiency,
                        "effective_soc_change": effective_soc_change
                    }
                    current_soc += effective_soc_change

                else:  # Gemiddelde prijs - geen actie
                    schedule[hour] = {
                        "action": "idle",
                        "setpoint": 0.0,
                        "reason": "Gemiddelde prijs - geen actie",
                        "price": price,
                        "efficiency": 1.0,
                        "effective_soc_change": 0.0
                    }

            # Bereken geschatte besparingen
            total_savings = self._calculate_savings(schedule, price_forecast)

            return {
                "schedule": schedule,
                "total_savings": total_savings,
                "final_soc": current_soc,
                "optimization_strategy": "price_based",
                "timestamp": dt.datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Fout bij batterij optimalisatie: {e}")
            return {"error": str(e)}

    def _calculate_savings(self, schedule: Dict, prices: Dict) -> float:
        """Berekent geschatte besparingen van het batterij schema"""
        try:
            savings = 0.0
            for hour, action in schedule.items():
                if action["action"] == "charge":
                    # Kosten van laden
                    savings -= action["setpoint"] * prices.get(hour, 0.25)
                elif action["action"] == "discharge":
                    # Besparingen van ontladen
                    savings += abs(action["setpoint"]) * prices.get(hour, 0.25)

            return round(savings, 2)
        except Exception as e:
            logging.error(f"Fout bij berekenen besparingen: {e}")
            return 0.0

    def _log_setpoint_change(self, setpoint: float, reason: str) -> None:
        """Logt een setpoint wijziging"""
        logging.info(f"BATTERY SETPOINT: {setpoint} kW - {reason}")

        # Hier zou je ook naar een database kunnen loggen
        # of een notificatie kunnen sturen

    def get_config_summary(self) -> Dict[str, Any]:
        """Geeft een samenvatting van de batterij configuratie"""
        return {
            "control_enabled": self.control_enabled,
            "battery_entity_id": self.battery_entity_id,
            "setpoint_entity_id": self.setpoint_entity_id,
            "min_soc": self.min_soc,
            "max_soc": self.max_soc,
            "max_charge_rate": self.max_charge_rate,
            "max_discharge_rate": self.max_discharge_rate,
            "battery_capacity": self.battery_capacity,
            "charge_efficiency_points": len(self.charge_efficiency_curve),
            "discharge_efficiency_points": len(self.discharge_efficiency_curve),
            "ha_url_configured": bool(self.ha_url),
            "ha_token_configured": bool(self.ha_token)
        }

    def get_efficiency_curves(self) -> Dict[str, Any]:
        """Geeft de efficiëntie curves voor laden en ontladen"""
        return {
            "charge_efficiency": self.charge_efficiency_curve,
            "discharge_efficiency": self.discharge_efficiency_curve,
            "battery_capacity": self.battery_capacity
        }

    def calculate_efficiency_at_soc(self, soc: float) -> Dict[str, float]:
        """Berekent de efficiëntie voor een gegeven SoC"""
        return {
            "charge_efficiency": self._get_efficiency_at_soc(soc, "charge"),
            "discharge_efficiency": self._get_efficiency_at_soc(soc, "discharge"),
            "soc": soc
        }
