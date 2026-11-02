from reader.spec.spec_types import DYNAMIC_DIMENSION, SpecComponentQuantity


def gen_init_quantities(name: str, quantity: SpecComponentQuantity):
    # torch cannot allocate a dynamic dimension, start it empty instead
    shape = tuple(0 if dim == DYNAMIC_DIMENSION else dim for dim in quantity.shape)
    return f"""
        self.quantities["{name}"] = torch.zeros(
            {shape}, device=self.device, dtype=self.dtype)
    """
