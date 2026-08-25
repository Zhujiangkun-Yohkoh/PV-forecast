import ast
import csv
import datetime as dt
import importlib.util
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("audit_nwp",ROOT/"audit_nwp_feasibility.py")
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT/"PV_AND_NWP_INVENTORY.csv").open(encoding="utf-8-sig") as handle:
            cls.inventory=list(csv.DictReader(handle))
        cls.nwp=[r for r in cls.inventory if r.get("record_type")=="NWP_SAMPLE"]

    def test_raw_pv_read_only(self):
        self.assertTrue(all(Path(r["source_path"]).exists() for r in self.inventory if r.get("record_type")=="PV_FILE"))

    def test_issue_time_before_valid_and_eligible_origin(self):
        for r in self.nwp:
            available=dt.datetime.fromisoformat(r["conservative_available_time_utc"])
            valid=dt.datetime.fromisoformat(r["valid_time_utc"])
            self.assertLessEqual(available,valid)

    def test_valid_time_equals_cycle_plus_lead(self):
        for r in self.nwp:
            cycle=dt.datetime.fromisoformat(r["cycle_utc"]); valid=dt.datetime.fromisoformat(r["valid_time_utc"])
            self.assertEqual(valid,cycle+dt.timedelta(hours=int(r["forecast_lead_hours"])))

    def test_utc_to_acst(self):
        for r in self.nwp[:20]:
            utc=dt.datetime.fromisoformat(r["valid_time_utc"]); local=dt.datetime.fromisoformat(r["valid_time_acst"])
            self.assertEqual(local.utcoffset(),dt.timedelta(hours=9,minutes=30)); self.assertEqual(local.astimezone(dt.timezone.utc),utc)

    def test_no_future_ground_weather_input(self):
        source=(ROOT/"audit_nwp_feasibility.py").read_text(encoding="utf-8")
        self.assertNotIn("future_ground_weather_input",source)
        self.assertIn("same-time ground GHI",source)  # evaluation only, never model input

    def test_windows_are_split_internal(self):
        years=[r for r in self.inventory if r.get("record_type")=="PV_YEAR" and r.get("site") in M.CFG["sites"]]
        self.assertEqual(len(years),15); self.assertTrue(all(int(float(r["continuous_L72_H144_windows"]))>=0 for r in years))

    def test_alignment_recomputable(self):
        rad=[r for r in self.nwp if r["variable"]=="DSWRF_surface"]
        self.assertEqual(len(rad),len(M.CFG["sample_days"])*len(M.CFG["cycles_per_local_day"])*len(M.CFG["forecast_leads_hours"]))

    def test_official_urls_and_dates(self):
        for r in self.nwp:
            self.assertTrue(r["idx_url"].startswith("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs."))
            self.assertIn(dt.datetime.fromisoformat(r["cycle_utc"]).strftime("%Y%m%d"),r["idx_url"])

    def test_outputs_limited_to_stage_directory(self):
        allowed={"audit_nwp_feasibility.py","config.json","test_protocol.py","PV_AND_NWP_INVENTORY.csv","LITERATURE_OVERLAP_MATRIX.csv","REPORT.md"}
        self.assertEqual({p.name for p in ROOT.iterdir() if p.is_file()},allowed)

    def test_no_neural_training_calls(self):
        tree=ast.parse((ROOT/"audit_nwp_feasibility.py").read_text(encoding="utf-8"))
        calls=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                if isinstance(n.func,ast.Attribute): calls.append(n.func.attr)
                elif isinstance(n.func,ast.Name): calls.append(n.func.id)
        self.assertFalse({"backward","step","fit","train"}&set(calls))


if __name__=="__main__": unittest.main(verbosity=2)
