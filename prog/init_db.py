#!/usr/bin/env python3
"""
Database initialisatie script voor DAO
Maakt de benodigde tabellen aan als deze niet bestaan
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_database():
    """Initialiseer de DAO database"""
    try:
        # Bepaal database pad
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
        db_path = data_dir / "day_ahead.db"

        # Maak data directory aan als deze niet bestaat
        data_dir.mkdir(exist_ok=True)

        logging.info(f"Database pad: {db_path}")

        # Maak verbinding met SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Maak basis tabellen aan
        create_tables(cursor)

        # Commit wijzigingen
        conn.commit()
        conn.close()

        logging.info("Database succesvol geïnitialiseerd")
        return True

    except Exception as e:
        logging.error(f"Database initialisatie fout: {e}")
        return False

def create_tables(cursor):
    """Maak de benodigde tabellen aan"""

    # Tabel voor energie data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS energy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            sensor_id TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT DEFAULT 'kWh',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabel voor prijs data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            price REAL NOT NULL,
            currency TEXT DEFAULT 'EUR',
            source TEXT DEFAULT 'nordpool',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabel voor batterij data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS battery_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            battery_id TEXT NOT NULL,
            soc REAL NOT NULL,
            power REAL DEFAULT 0,
            temperature REAL DEFAULT 20,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabel voor optimalisatie resultaten
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            period_start DATETIME NOT NULL,
            period_end DATETIME NOT NULL,
            total_cost REAL NOT NULL,
            total_savings REAL DEFAULT 0,
            strategy TEXT DEFAULT 'minimize_cost',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Maak indexes aan voor betere performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_energy_data_timestamp ON energy_data(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_energy_data_sensor ON energy_data(sensor_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON price_data(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_battery_data_timestamp ON battery_data(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_battery_data_battery ON battery_data(battery_id)")

    logging.info("Database tabellen aangemaakt")

if __name__ == "__main__":
    success = init_database()
    if success:
        print("✅ Database succesvol geïnitialiseerd")
        sys.exit(0)
    else:
        print("❌ Database initialisatie mislukt")
        sys.exit(1)
