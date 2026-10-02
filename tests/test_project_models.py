from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from magicborder.io_utils import (
    load_annotation,
    load_project,
    save_annotation,
    save_project,
)
from magicborder.models import (
    ANGLE_LABEL_COLOR,
    ANGLE_LINE_COLOR,
    CONTOUR_LINE_COLOR,
    PROJECT_FORMAT_VERSION,
    SEGMENT_LABEL_COLOR,
    SEGMENT_LINE_COLOR,
    Annotation,
    ImageCalibration,
    Point,
    ProjectAngleMeasurement,
    ProjectDocument,
    ProjectImageMeasurements,
    ProjectImageRecord,
    ProjectInfo,
    ProjectMeasurementAssessment,
    ProjectSegmentMeasurement,
)


class ProjectModelsTest(unittest.TestCase):
    def test_project_round_trip_preserves_image_annotation(self) -> None:
        annotation = Annotation(
            image_path="images/leaf.png",
            image_width=10,
            image_height=8,
            points=[Point(1, 1), Point(8, 1), Point(6, 6)],
            line_color="#123ABC",
        )
        project = ProjectDocument(
            name="leaves",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    image_width=10,
                    image_height=8,
                    annotation=annotation,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "leaves.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(payload["version"], PROJECT_FORMAT_VERSION)
        self.assertIn("file", record_payload)
        self.assertIn("contour", record_payload)
        self.assertIn("location", record_payload)
        self.assertIn("details", record_payload)
        self.assertNotIn("metadata", record_payload)
        self.assertNotIn("path", record_payload)
        self.assertEqual(record_payload["file"]["id"], "leaf-1")
        self.assertEqual(record_payload["file"]["path"], "images/leaf.png")
        self.assertIsNotNone(record_payload["contour"]["annotation"])
        self.assertEqual(
            record_payload["contour"]["annotation"]["line_color"],
            "#123abc",
        )
        self.assertEqual(loaded_project.name, "leaves")
        self.assertEqual(len(loaded_project.images), 1)
        loaded_record = loaded_project.images[0]
        self.assertEqual(loaded_record.relative_path, "images/leaf.png")
        self.assertIsNotNone(loaded_record.annotation)
        self.assertEqual(loaded_record.annotation.image_width, 10)
        self.assertEqual(len(loaded_record.annotation.points), 3)
        self.assertEqual(loaded_record.annotation.line_color, "#123abc")

    def test_project_round_trip_preserves_image_calibration(self) -> None:
        calibration = ImageCalibration(
            start=Point(1, 2),
            end=Point(121, 2),
            length_mm=10,
        )
        project = ProjectDocument(
            name="calibration",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    calibration=calibration,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "calibration.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(
            record_payload["calibration"],
            {
                "start": {"x": 1.0, "y": 2.0},
                "end": {"x": 121.0, "y": 2.0},
                "length_mm": 10.0,
            },
        )
        loaded_calibration = loaded_project.images[0].calibration
        self.assertIsNotNone(loaded_calibration)
        self.assertEqual(loaded_calibration.start, Point(1, 2))
        self.assertEqual(loaded_calibration.end, Point(121, 2))
        self.assertEqual(loaded_calibration.length_mm, 10.0)
        self.assertEqual(loaded_calibration.pixel_length(), 120.0)
        self.assertEqual(loaded_calibration.pixels_per_mm(), 12.0)

    def test_project_round_trip_preserves_image_angle_measurements(self) -> None:
        project = ProjectDocument(
            name="measurements",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    measurements=ProjectImageMeasurements(
                        angles=[
                            ProjectAngleMeasurement(
                                id="angle-1",
                                first=Point(2, 10),
                                vertex=Point(2, 2),
                                second=Point(10, 2),
                                name="Контрольный угол",
                                line_color="#16A34A",
                                label_color="#15803D",
                                note="контрольный угол",
                            )
                        ]
                    ),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "measurements.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(
            record_payload["measurements"]["angles"],
            [
                {
                    "id": "angle-1",
                    "first": {"x": 2.0, "y": 10.0},
                    "vertex": {"x": 2.0, "y": 2.0},
                    "second": {"x": 10.0, "y": 2.0},
                    "name": "Контрольный угол",
                    "line_color": "#16a34a",
                    "label_color": "#15803d",
                    "note": "контрольный угол",
                }
            ],
        )
        loaded_angle = loaded_project.images[0].measurements.angles[0]
        self.assertEqual(loaded_angle.id, "angle-1")
        self.assertEqual(loaded_angle.first, Point(2, 10))
        self.assertEqual(loaded_angle.vertex, Point(2, 2))
        self.assertEqual(loaded_angle.second, Point(10, 2))
        self.assertEqual(loaded_angle.name, "Контрольный угол")
        self.assertEqual(loaded_angle.line_color, "#16a34a")
        self.assertEqual(loaded_angle.label_color, "#15803d")
        self.assertEqual(loaded_angle.note, "контрольный угол")

    def test_project_round_trip_preserves_image_segment_measurements(self) -> None:
        project = ProjectDocument(
            name="segments",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    measurements=ProjectImageMeasurements(
                        segments=[
                            ProjectSegmentMeasurement(
                                id="segment-1",
                                start=Point(10, 20),
                                end=Point(80, 20),
                                name="Контрольный отрезок",
                                line_color="#EA580C",
                                label_color="#C2410C",
                                start_label="A",
                                end_label="B",
                                note="измерить повторно",
                            )
                        ],
                        extra_groups={"future": [{"kept": True}]},
                    ),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "segments.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(
            record_payload["measurements"]["segments"],
            [
                {
                    "id": "segment-1",
                    "name": "Контрольный отрезок",
                    "start": {"x": 10.0, "y": 20.0},
                    "end": {"x": 80.0, "y": 20.0},
                    "line_color": "#ea580c",
                    "label_color": "#c2410c",
                    "start_label": "A",
                    "end_label": "B",
                    "note": "измерить повторно",
                }
            ],
        )
        self.assertEqual(record_payload["measurements"]["future"], [{"kept": True}])
        loaded_measurements = loaded_project.images[0].measurements
        loaded_segment = loaded_measurements.segments[0]
        self.assertEqual(loaded_segment.id, "segment-1")
        self.assertEqual(loaded_segment.start, Point(10, 20))
        self.assertEqual(loaded_segment.end, Point(80, 20))
        self.assertEqual(loaded_segment.name, "Контрольный отрезок")
        self.assertEqual(loaded_segment.line_color, "#ea580c")
        self.assertEqual(loaded_segment.label_color, "#c2410c")
        self.assertEqual(loaded_segment.start_label, "A")
        self.assertEqual(loaded_segment.end_label, "B")
        self.assertEqual(loaded_segment.note, "измерить повторно")
        self.assertEqual(loaded_measurements.extra_groups, {"future": [{"kept": True}]})

    def test_line_colors_default_and_invalid_values_are_backward_compatible(
        self,
    ) -> None:
        annotation_payload = {
            "image_path": "images/leaf.png",
            "image_size": {"width": 10, "height": 8},
            "points": [
                {"x": 1, "y": 1},
                {"x": 8, "y": 1},
                {"x": 6, "y": 6},
            ],
        }
        self.assertEqual(
            Annotation.from_dict(annotation_payload).line_color,
            CONTOUR_LINE_COLOR,
        )
        invalid_annotation = Annotation.from_dict(annotation_payload)
        invalid_annotation.line_color = "still-not-a-color"
        self.assertEqual(
            invalid_annotation.to_dict()["line_color"],
            CONTOUR_LINE_COLOR,
        )
        annotation_payload["line_color"] = "not-a-color"
        self.assertEqual(
            Annotation.from_dict(annotation_payload).line_color,
            CONTOUR_LINE_COLOR,
        )

        angle_payload = {
            "id": "angle-1",
            "first": {"x": 2, "y": 10},
            "vertex": {"x": 2, "y": 2},
            "second": {"x": 10, "y": 2},
        }
        self.assertEqual(
            ProjectAngleMeasurement.from_dict(angle_payload).line_color,
            ANGLE_LINE_COLOR,
        )
        self.assertEqual(
            ProjectAngleMeasurement.from_dict(angle_payload).label_color,
            ANGLE_LABEL_COLOR,
        )
        invalid_angle = ProjectAngleMeasurement.from_dict(angle_payload)
        invalid_angle.line_color = "still-not-a-color"
        invalid_angle.label_color = "still-not-a-color"
        self.assertEqual(invalid_angle.to_dict()["line_color"], ANGLE_LINE_COLOR)
        self.assertEqual(invalid_angle.to_dict()["label_color"], ANGLE_LABEL_COLOR)
        angle_payload["line_color"] = "#12"
        angle_payload["label_color"] = "#12"
        self.assertEqual(
            ProjectAngleMeasurement.from_dict(angle_payload).line_color,
            ANGLE_LINE_COLOR,
        )
        self.assertEqual(
            ProjectAngleMeasurement.from_dict(angle_payload).label_color,
            ANGLE_LABEL_COLOR,
        )

        segment_payload = {
            "id": "segment-1",
            "start": {"x": 10, "y": 20},
            "end": {"x": 80, "y": 20},
        }
        self.assertEqual(
            ProjectSegmentMeasurement.from_dict(segment_payload).line_color,
            SEGMENT_LINE_COLOR,
        )
        self.assertEqual(
            ProjectSegmentMeasurement.from_dict(segment_payload).label_color,
            SEGMENT_LABEL_COLOR,
        )
        invalid_segment = ProjectSegmentMeasurement.from_dict(segment_payload)
        invalid_segment.line_color = "still-not-a-color"
        invalid_segment.label_color = "still-not-a-color"
        self.assertEqual(invalid_segment.to_dict()["line_color"], SEGMENT_LINE_COLOR)
        self.assertEqual(invalid_segment.to_dict()["label_color"], SEGMENT_LABEL_COLOR)
        segment_payload["line_color"] = "orange"
        segment_payload["label_color"] = "orange"
        self.assertEqual(
            ProjectSegmentMeasurement.from_dict(segment_payload).line_color,
            SEGMENT_LINE_COLOR,
        )
        self.assertEqual(
            ProjectSegmentMeasurement.from_dict(segment_payload).label_color,
            SEGMENT_LABEL_COLOR,
        )

    def test_annotation_file_round_trip_preserves_line_color(self) -> None:
        annotation = Annotation(
            image_path="images/leaf.png",
            image_width=10,
            image_height=8,
            points=[Point(1, 1), Point(8, 1), Point(6, 6)],
            line_color="#A855F7",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            annotation_path = Path(temp_dir) / "leaf.json"
            save_annotation(annotation_path, annotation)
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            loaded_annotation = load_annotation(annotation_path)

        self.assertEqual(payload["line_color"], "#a855f7")
        self.assertEqual(loaded_annotation.line_color, "#a855f7")

    def test_project_round_trip_preserves_measurement_oiv_assessments(self) -> None:
        project = ProjectDocument(
            name="assessments",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    measurements=ProjectImageMeasurements(
                        angles=[
                            ProjectAngleMeasurement(
                                id="angle-1",
                                first=Point(2, 10),
                                vertex=Point(2, 2),
                                second=Point(10, 2),
                                assessment=ProjectMeasurementAssessment(
                                    system="OIV",
                                    code="OIV 607",
                                ),
                            )
                        ],
                        segments=[
                            ProjectSegmentMeasurement(
                                id="segment-1",
                                start=Point(10, 20),
                                end=Point(80, 20),
                                assessment=ProjectMeasurementAssessment(
                                    system="OIV",
                                    code="OIV 601",
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "assessments.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        image_payload = payload["images"][0]
        self.assertEqual(
            image_payload["measurements"]["angles"][0]["assessment"],
            {"system": "OIV", "code": "OIV 607"},
        )
        self.assertEqual(
            image_payload["measurements"]["segments"][0]["assessment"],
            {"system": "OIV", "code": "OIV 601"},
        )
        self.assertNotIn("score", image_payload["measurements"]["angles"][0])
        self.assertNotIn("score", image_payload["measurements"]["segments"][0])

        loaded_measurements = loaded_project.images[0].measurements
        self.assertEqual(loaded_measurements.angles[0].assessment.code, "OIV 607")
        self.assertEqual(loaded_measurements.segments[0].assessment.code, "OIV 601")

    def test_project_loads_legacy_segment_measurement_without_name_and_note(
        self,
    ) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "legacy_segments",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "measurements": {
                            "segments": [
                                {
                                    "id": "segment-1",
                                    "start": {"x": 10.0, "y": 20.0},
                                    "end": {"x": 80.0, "y": 20.0},
                                    "start_label": "A",
                                    "end_label": "B",
                                }
                            ]
                        },
                    }
                ],
            }
        )

        segment = project.images[0].measurements.segments[0]

        self.assertEqual(segment.name, "")
        self.assertEqual(segment.note, "")
        self.assertEqual(segment.start_label, "A")
        self.assertEqual(segment.end_label, "B")
        self.assertNotIn("name", segment.to_dict())
        self.assertEqual(segment.to_dict()["note"], "")

    def test_project_skips_corrupt_segment_measurements(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "legacy_segments",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "measurements": {
                            "angles": [],
                            "segments": [
                                {
                                    "id": "segment-good",
                                    "start": {"x": 0, "y": 0},
                                    "end": {"x": 10, "y": 0},
                                    "start_label": "A",
                                    "end_label": "B",
                                },
                                {
                                    "id": "segment-bad",
                                    "start": {"x": 1, "y": 1},
                                    "end": {"x": 1, "y": 1},
                                },
                                {
                                    "id": "segment-broken",
                                    "start": {"x": "x", "y": 0},
                                    "end": {"x": 2, "y": 0},
                                },
                            ],
                        },
                    }
                ],
            }
        )

        segments = project.images[0].measurements.segments

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].id, "segment-good")

    def test_project_loads_legacy_angle_measurement_without_name(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "legacy_angles",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "measurements": {
                            "angles": [
                                {
                                    "id": "angle-1",
                                    "first": {"x": 2.0, "y": 10.0},
                                    "vertex": {"x": 2.0, "y": 2.0},
                                    "second": {"x": 10.0, "y": 2.0},
                                    "note": "старый угол",
                                }
                            ]
                        },
                    }
                ],
            }
        )

        angle = project.images[0].measurements.angles[0]

        self.assertEqual(angle.name, "")
        self.assertEqual(angle.note, "старый угол")
        self.assertNotIn("name", angle.to_dict())

    def test_project_keeps_corrupt_calibration_payload_without_breaking_load(
        self,
    ) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "broken_calibration",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "contour": {},
                        "location": {},
                        "details": {},
                        "calibration": {
                            "start": {"x": 1, "y": 1},
                            "end": {"x": 1, "y": 1},
                            "length_mm": 10,
                        },
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertIsNone(record.calibration)
        self.assertIsNotNone(record.calibration_error)
        self.assertEqual(
            record.to_dict()["calibration"],
            {
                "start": {"x": 1, "y": 1},
                "end": {"x": 1, "y": 1},
                "length_mm": 10,
            },
        )

    def test_project_loads_images_without_measurements_as_empty_collection(
        self,
    ) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "old_project",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "contour": {},
                        "location": {},
                        "details": {},
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertEqual(record.measurements.angles, [])
        self.assertEqual(record.measurements.segments, [])
        self.assertNotIn("measurements", record.to_dict())

    def test_project_info_round_trip_uses_grouped_json(self) -> None:
        project = ProjectDocument(
            name="leaves",
            images=[],
            project_info=ProjectInfo(general_info="Полевой эксперимент"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "leaves.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        self.assertEqual(
            payload["project_info"], {"general_info": "Полевой эксперимент"}
        )
        self.assertEqual(
            loaded_project.project_info.general_info, "Полевой эксперимент"
        )

    def test_project_info_defaults_for_old_json(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "old",
                "images": [],
            }
        )

        self.assertEqual(project.project_info.general_info, "")

    def test_project_keeps_corrupt_annotation_payload(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "broken",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "contour": {"annotation": {"points": []}},
                        "location": {},
                        "details": {},
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertIsNone(record.annotation)
        self.assertIsNotNone(record.annotation_error)
        self.assertEqual(record.to_dict()["contour"]["annotation"], {"points": []})

    def test_project_image_metadata_round_trip_uses_grouped_json(self) -> None:
        project = ProjectDocument(
            name="metadata",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    metadata={
                        "added_at": "2026-05-03T10:20:30+03:00",
                        "captured_at": "2026-05-04T08:00:00+03:00",
                        "humidity": "55",
                        "diagnosis": "leaf_blight",
                    },
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "metadata.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(
            record_payload["file"]["added_at"], "2026-05-03T10:20:30+03:00"
        )
        self.assertEqual(
            record_payload["file"]["captured_at"], "2026-05-04T08:00:00+03:00"
        )
        self.assertEqual(record_payload["location"]["humidity"], "55")
        self.assertEqual(record_payload["details"]["diagnosis"], "leaf_blight")
        self.assertNotIn("metadata", record_payload)

        metadata = loaded_project.images[0].metadata
        self.assertEqual(metadata["added_at"], "2026-05-03T10:20:30+03:00")
        self.assertEqual(metadata["captured_at"], "2026-05-04T08:00:00+03:00")
        self.assertEqual(metadata["humidity"], "55")
        self.assertEqual(metadata["diagnosis"], "leaf_blight")

    def test_project_image_group_defaults(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "group_defaults",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "contour": {},
                        "location": {},
                        "details": {},
                    }
                ],
            }
        )

        metadata = project.images[0].metadata

        self.assertEqual(metadata["added_at"], "")
        self.assertEqual(metadata["diagnosis"], "Не указано")
        self.assertEqual(metadata["notes"], "")

    def test_old_flat_project_records_are_migrated(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "name": "old",
                "images": [
                    {
                        "id": "record-id",
                        "path": "images/leaf.png",
                        "display_name": "leaf.png",
                    }
                ],
            }
        )

        self.assertEqual([record.id for record in project.images], ["record-id"])
        self.assertEqual(project.images[0].relative_path, "images/leaf.png")
        self.assertEqual(project.images[0].display_name, "leaf.png")


if __name__ == "__main__":
    unittest.main()
