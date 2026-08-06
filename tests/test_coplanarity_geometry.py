import unittest

import numpy as np
import pandas as pd

from Pipelines.TrackML_Example.LightningModules.Processing.utils.coplanarity import (
    assign_layers_from_geometry,
    build_geometry_layer_map,
    paper_plane_finding,
    roc_curve_points,
)


class CoplanarityGeometryTests(unittest.TestCase):
    def setUp(self):
        rows = []
        module_id = 1
        for volume_id, layer_id, radius in ((8, 2, 32.0), (13, 2, 260.0), (17, 4, 1020.0)):
            for z in (-500.0, 0.0, 500.0):
                rows.append({
                    "volume_id": volume_id, "layer_id": layer_id,
                    "module_id": module_id, "cx": radius, "cy": 0.0, "cz": z,
                })
                module_id += 1
        # Endcap-like disk: broad radius and nearly fixed z.
        for radius in (60.0, 100.0, 140.0):
            rows.append({
                "volume_id": 7, "layer_id": 2, "module_id": module_id,
                "cx": radius, "cy": 0.0, "cz": -1500.0,
            })
            module_id += 1
        self.detector = pd.DataFrame(rows)

    def test_geometry_map_discovers_barrels_and_orders_by_radius(self):
        mapping = build_geometry_layer_map(self.detector)
        self.assertEqual(mapping, {(8, 2): 1, (13, 2): 2, (17, 4): 3})
        self.assertNotIn((7, 2), mapping)

    def test_module_indices_map_to_exact_geometry_layers(self):
        assigned = assign_layers_from_geometry([0, 3, 6, 9], self.detector)
        np.testing.assert_array_equal(assigned, [1, 2, 3, 0])

    def test_plane_finder_uses_outermost_populated_layers(self):
        xyz = []
        layers = []
        for layer, radius in ((1, 30.0), (2, 70.0), (3, 120.0)):
            for phi in (0.0, 0.01):
                xyz.append([radius * np.cos(phi), radius * np.sin(phi), 0.0])
                layers.append(layer)
        result = paper_plane_finding(
            np.asarray(xyz), np.asarray(layers),
            ds_seed=0.5, dw_seed=10.0, ds_iter=0.5, dw_iter=10.0,
            ds_final=0.1, dw_final=10.0,
            min_hits_per_layer=2, min_layers_with_2hits=2,
        )
        self.assertTrue(result["found"])
        self.assertEqual(result["outer_layer"], 3)
        self.assertEqual(result["second_layer"], 2)

    def test_tied_scores_have_random_auc(self):
        *_, auc = roc_curve_points([1.0, 1.0], [1.0, 1.0])
        self.assertAlmostEqual(auc, 0.5)


if __name__ == "__main__":
    unittest.main()
