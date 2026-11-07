import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import SimulationConfig
from data_pipeline import generate_dataset
from validate_dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def test_generated_dataset_passes_integrity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.sqlite"
            generate_dataset(
                path,
                samples_per_class=30,
                seed=11,
                config=SimulationConfig(
                    time_ms=80.0,
                    dt_ms=0.2,
                    current_min=12.0,
                    current_max=25.0,
                    minimum_spikes=1,
                ),
            )
            connections = []
            connect = sqlite3.connect

            def tracked_connect(*args, **kwargs):
                connection = connect(*args, **kwargs)
                connections.append(connection)
                self.addCleanup(connection.close)
                return connection

            # Retain the connections so garbage collection cannot mask leaks.
            with patch("validate_dataset.sqlite3.connect", side_effect=tracked_connect):
                report = validate_dataset(path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["total_samples"], 90)
            self.assertEqual(len(connections), 2)
            for connection in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_connection_is_closed_when_validation_query_fails(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        metadata = {"simulation": {"window_size": 400}, "samples_per_class": 30}
        with (
            patch("validate_dataset.read_metadata", return_value=metadata),
            patch("validate_dataset.sqlite3.connect", return_value=connection),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "no such table"):
                validate_dataset(Path("unused.sqlite"))
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
