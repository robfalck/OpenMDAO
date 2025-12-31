from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from openmdao.specs.indices_spec import SrcIndicesSpec


class ConnectionSpec(BaseModel):
    """
    Specification for a connection between variables.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    src: str = Field(description="Source variable name")
    tgt: str = Field(description="Target variable name")
    src_indices: SrcIndicesSpec = Field(
        default_factory=lambda: SrcIndicesSpec(value=None),
        description="Indices into source variable"
    )


if __name__ == "__main__":
    from openmdao.api import slicer
    from openmdao.specs import ConstraintSpec, IndicesSpec
    
    print("=" * 60)
    print("IndicesSpec Examples (for constraints/objectives)")
    print("=" * 60)
    
    # Constraint on all elements
    con1 = ConstraintSpec(
        name="con1",
        lower=0.0,
        indices=IndicesSpec(value=None)
    )
    print(f"\n1. All elements: {con1.indices}")
    
    # Constraint on specific indices using slicer (returns a tuple)
    con2 = ConstraintSpec(
        name="con2", 
        upper=10.0,
        indices=IndicesSpec(value=slicer[::2])  # This is just (slice(None, None, 2),)
    )
    print(f"2. Every other element: {con2.indices}")
    
    # Constraint on explicit list of indices
    con3 = ConstraintSpec(
        name="con3",
        equals=5.0,
        indices=IndicesSpec(value=[0, 5, 10, 15])
    )
    print(f"3. Explicit indices: {con3.indices}")
    
    # Test applying indices
    test_array = np.arange(20)
    print(f"\nTest array: {test_array}")
    print(f"con1 applied: {con1.indices.apply(test_array)}")
    print(f"con2 applied: {con2.indices.apply(test_array)}")
    print(f"con3 applied: {con3.indices.apply(test_array)}")
    
    # Serialization
    con2_dict = con2.model_dump()
    print(f"\nSerialized constraint: {con2_dict}")
    con2_restored = ConstraintSpec.model_validate(con2_dict)
    print(f"Restored constraint: {con2_restored}")
    print(f"Restored con2 applied: {con2_restored.indices.apply(test_array)}")
    
    print("\n" + "=" * 60)
    print("SrcIndicesSpec Examples (for connections)")
    print("=" * 60)
    
    # No indexing - 1:1 connection
    conn1 = ConnectionSpec(
        src="comp1.x",
        tgt="comp2.y",
        src_indices=SrcIndicesSpec(value=None)
    )
    print(f"\n1. Direct connection: {conn1.src_indices}")
    
    # Shaped indexing with slicer (returns a tuple)
    conn2 = ConnectionSpec(
        src="comp1.x",
        tgt="comp2.y",
        src_indices=SrcIndicesSpec(
            value=slicer[1:5, ::2],  # This is (slice(1, 5), slice(None, None, 2))
            flat=False,
            src_shape=(10, 10)
        )
    )
    print(f"2. Shaped indexing: {conn2.src_indices}")
    
    # Flat indexing with explicit indices
    conn3 = ConnectionSpec(
        src="comp1.x",
        tgt="comp2.y",
        src_indices=SrcIndicesSpec(
            value=[0, 10, 20, 30],
            flat=True
        )
    )
    print(f"3. Flat explicit indices: {conn3.src_indices}")
    
    # Test applying src_indices
    src_array = np.arange(100).reshape(10, 10)
    print(f"\nSource array shape: {src_array.shape}")
    print(f"conn2 applied shape: {conn2.src_indices.apply(src_array).shape}")
    print(f"conn3 applied: {conn3.src_indices.apply(src_array)}")
    
    # Serialization
    conn2_dict = conn2.model_dump()
    print(f"\nSerialized connection: {conn2_dict}")
    conn2_restored = ConnectionSpec.model_validate(conn2_dict)
    print(f"Restored connection: {conn2_restored}")
    print(f"Restored conn2 applied: {conn2_restored.src_indices.apply(src_array).shape}")

