"""Components take their dtype and device from PyTorch's defaults at construction."""
# pylint: disable=missing-class-docstring, missing-function-docstring

import unittest

import torch

from aicon_base.components.base_components import Component


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_torch_type(self):
        """
        Test if the dtype of a Component changes as desired after changing
        the pytorch's global default dtype.
        """

        def handle_connection(device, dtype, mockbuild):
            print(device, dtype, mockbuild)

        connections = {"con1": handle_connection}

        dtype = torch.get_default_dtype()

        component_1 = Component("meaningless_name", connections)
        self.assertEqual(component_1.dtype, dtype)
        self.assertEqual(component_1.device, torch.device("cpu"))

        # Change the default dtype of pytorch
        torch.set_default_dtype(torch.float64)
        component_2 = Component("meaningless_name", connections)
        self.assertEqual(component_1.dtype, dtype)
        self.assertEqual(component_2.dtype, torch.float64)
        self.assertEqual(component_2.device, torch.device("cpu"))

        # Change the default dtype of pytorch
        torch.set_default_dtype(torch.float32)
        component_3 = Component("meaningless_name", connections)
        self.assertEqual(component_1.dtype, dtype)
        self.assertEqual(component_2.dtype, torch.float64)
        self.assertEqual(component_3.dtype, torch.float32)
        self.assertEqual(component_3.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
