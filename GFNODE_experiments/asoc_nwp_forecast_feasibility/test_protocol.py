import ast
import csv
import datetime as dt
import importlib.util
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
        cls.messages=[r for r in cls.inventory if r.get("record_type")=="NWP_PILOT_MESSAGE"]
        cls.mappings=[r for r in cls.inventory if r.get("record_type")=="NWP_ORIGIN_MAPPING"]
        cls.objects=[r for r in cls.inventory if r.get("record_type")=="NWP_PILOT_OBJECT"]
        cls.segments=[r for r in cls.inventory if r.get("record_type")=="PV_LEGAL_SEGMENT"]
        cls.report=(ROOT/"REPORT.md").read_text(encoding="utf-8")

    def test_completed_cycle_available_before_origin(self):
        for r in self.mappings:
            cycle=dt.datetime.fromisoformat(r["selected_cycle_utc"]); origin=dt.datetime.fromisoformat(r["forecast_origin_utc"])
            self.assertLessEqual(cycle+dt.timedelta(hours=6),origin)

    def test_selected_cycle_never_from_origin_future(self):
        for r in self.mappings:
            self.assertLess(dt.datetime.fromisoformat(r["selected_cycle_utc"]),dt.datetime.fromisoformat(r["forecast_origin_utc"]))

    def test_valid_time_equals_cycle_plus_lead(self):
        for r in self.messages:
            cycle=dt.datetime.fromisoformat(r["cycle_utc"]); valid=dt.datetime.fromisoformat(r["valid_time_utc"])
            self.assertEqual(valid,cycle+dt.timedelta(hours=float(r["forecast_lead_hours"])))

    def test_forecast_age_is_origin_minus_cycle(self):
        for r in self.mappings:
            origin=dt.datetime.fromisoformat(r["forecast_origin_utc"]); cycle=dt.datetime.fromisoformat(r["selected_cycle_utc"])
            self.assertAlmostEqual(float(r["forecast_age_hours"]),(origin-cycle).total_seconds()/3600,places=8)

    def test_entire_h144_trajectory_uses_one_cycle(self):
        self.assertTrue(self.mappings)
        self.assertTrue(all(r["trajectory_cycle_policy"]=="SINGLE_SELECTED_CYCLE" for r in self.mappings))
        self.assertTrue(all("f006..f024" in r["source_object"] for r in self.mappings if r["nwp_valid"]=="True"))

    def test_missing_objects_only_fall_back_to_earlier_cycles(self):
        for r in self.mappings:
            origin=dt.datetime.fromisoformat(r["forecast_origin_utc"]); selected=dt.datetime.fromisoformat(r["selected_cycle_utc"])
            rounded=origin.replace(hour=(origin.hour//6)*6,minute=0,second=0,microsecond=0)-dt.timedelta(hours=6)
            self.assertEqual(selected,rounded-dt.timedelta(hours=6*int(r["fallback_cycles"])))

    def test_pv_windows_remain_inside_segment_and_split(self):
        splits={k:(dt.datetime.fromisoformat(v[0]),dt.datetime.fromisoformat(v[1])) for k,v in M.CFG["preferred_splits"].items()}
        for r in self.segments:
            start=dt.datetime.fromisoformat(r["period_start"]); end=dt.datetime.fromisoformat(r["period_end"])
            self.assertGreaterEqual(start,splits[r["split"]][0]); self.assertLessEqual(end,splits[r["split"]][1])
            self.assertGreaterEqual(int(r["segment_length_5min"]),M.WINDOW)

    def test_test_uses_all_2023_legal_segments(self):
        test_segments=[r for r in self.segments if r["split"]=="test"]
        self.assertGreater(len(test_segments),4)
        self.assertIn("Test: 2023-01-01 00:00–2023-12-31 23:55",self.report)
        self.assertNotIn("longest common valid 2023 block",self.report)

    def test_config_and_report_dates_match(self):
        for split,(start,end) in M.CFG["preferred_splits"].items():
            self.assertIn(f"{split.capitalize()}: {start[:-3]}",self.report)
            self.assertIn(end[:-3],self.report)

    def test_grib_statistical_semantics_are_used(self):
        ds=[r for r in self.messages if r["variable"]=="DSWRF_surface"]
        ap=[r for r in self.messages if r["variable"]=="APCP_surface"]
        self.assertTrue(ds and ap); self.assertTrue(all(r["stepType"]=="avg" for r in ds)); self.assertTrue(all(r["stepType"]=="accum" for r in ap))
        example=ap[0]; duration=float(example["endStep"])-float(example["startStep"])
        self.assertGreater(duration,0)
        self.assertAlmostEqual(M.aligned_cycle_value([dict(example,value=float(example["value"]))],"APCP_surface",float(example["endStep"])),float(example["value"])/duration)

    def test_future_ground_weather_is_audit_only(self):
        source=(ROOT/"audit_nwp_feasibility.py").read_text(encoding="utf-8")
        self.assertIn("Ground GHI is audit/label-side information only",source); self.assertNotIn("future_ground_weather_input",source)

    def test_source_pv_size_and_mtime_unchanged(self):
        rows=[r for r in self.inventory if r.get("record_type")=="PV_FILE"]
        self.assertTrue(rows)
        for r in rows:
            stat=Path(r["source_path"]).stat()
            self.assertEqual(stat.st_size,int(r["file_size_bytes"])); self.assertEqual(stat.st_mtime_ns,int(r["source_mtime_ns"]))

    def test_no_neural_training_calls(self):
        tree=ast.parse((ROOT/"audit_nwp_feasibility.py").read_text(encoding="utf-8")); calls=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Attribute): calls.append(node.func.attr)
                elif isinstance(node.func,ast.Name): calls.append(node.func.id)
        self.assertFalse({"backward","optimizer","fit","train"}&set(calls))

    def test_outputs_remain_exactly_six_files(self):
        allowed={"audit_nwp_feasibility.py","config.json","test_protocol.py","PV_AND_NWP_INVENTORY.csv","LITERATURE_OVERLAP_MATRIX.csv","REPORT.md"}
        self.assertEqual({p.name for p in ROOT.iterdir() if p.is_file()},allowed)

    def test_real_continuous_pilot_and_official_urls(self):
        self.assertEqual(len(self.mappings),7*24*12); self.assertEqual(len(self.objects),8*4*19)
        self.assertTrue(all(r["source_object"].startswith("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.") for r in self.objects))

    def test_utc_to_acst_conversion(self):
        for r in self.messages[:50]:
            utc=dt.datetime.fromisoformat(r["valid_time_utc"]); local=dt.datetime.fromisoformat(r["valid_time_acst"])
            self.assertEqual(local.utcoffset(),dt.timedelta(hours=9,minutes=30)); self.assertEqual(local.astimezone(dt.timezone.utc),utc)

    def test_origin_alignment_is_recomputable(self):
        for r in self.mappings[:100]:
            origin=dt.datetime.fromisoformat(r["forecast_origin_utc"]); cycle=dt.datetime.fromisoformat(r["selected_cycle_utc"])
            age=(origin-cycle).total_seconds()/3600
            self.assertTrue(6<=age<12); self.assertEqual(int(r["expected_nwp_points"]),12*12*7)


if __name__=="__main__": unittest.main(verbosity=2)
