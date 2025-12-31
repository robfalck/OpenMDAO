from collections.abc import Sequence
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np


def serialize_index_tuple(idx_tuple: tuple) -> list:
    """Convert index tuple to JSON-serializable format."""
    if not isinstance(idx_tuple, tuple):
        idx_tuple = (idx_tuple,)
    
    result = []
    for elem in idx_tuple:
        if elem is Ellipsis:
            result.append({'type': 'ellipsis'})
        elif elem is np.newaxis or elem is None:
            result.append({'type': 'newaxis'})
        elif isinstance(elem, slice):
            result.append({
                'type': 'slice',
                'start': elem.start,
                'stop': elem.stop,
                'step': elem.step
            })
        elif isinstance(elem, (int, np.integer)):
            result.append({'type': 'int', 'value': int(elem)})
        elif isinstance(elem, np.ndarray):
            result.append({
                'type': 'array',
                'value': elem.tolist(),
                'dtype': str(elem.dtype)
            })
        else:
            raise ValueError(f"Cannot serialize index element: {elem}")
    return result


def deserialize_index_tuple(data: list) -> tuple:
    """Convert serialized format back to index tuple."""
    result = []
    for elem in data:
        if elem['type'] == 'ellipsis':
            result.append(Ellipsis)
        elif elem['type'] == 'newaxis':
            result.append(np.newaxis)
        elif elem['type'] == 'slice':
            result.append(slice(elem['start'], elem['stop'], elem['step']))
        elif elem['type'] == 'int':
            result.append(elem['value'])
        elif elem['type'] == 'array':
            result.append(np.array(elem['value'], dtype=elem['dtype']))
    return tuple(result)


class IndicesSpec(BaseModel):
    """
    Specification for constraint/objective indices in OpenMDAO.
    
    Used when defining constraints, objectives, or other operations that
    need to select specific elements from a variable's full array.
    
    Can be:
    - None: Use all elements
    - tuple: Index tuple (from slicer or index_exp)
    - ndarray: Explicit integer indices
    - List of ints: Specific indices
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    value: None | Sequence[int] | np.ndarray = Field(
        default=None,
        description="Indices specification"
    )

    flat: bool = Field(
        default=False,
        description="If True, indices index into flattened source"
    )
    
    @field_validator('value', mode='before')
    @classmethod
    def validate_indices(cls, v):
        """Convert various inputs to appropriate index type."""
        if v is None:
            return None
        
        # Already a tuple (from slicer)
        if isinstance(v, tuple):
            return v
        
        # Dict from serialization
        if isinstance(v, dict):
            if v.get('type') == 'tuple':
                return deserialize_index_tuple(v['value'])
            elif v.get('type') == 'array':
                return np.array(v['value'])
        
        # Numpy array of integers
        if isinstance(v, np.ndarray):
            if not np.issubdtype(v.dtype, np.integer):
                raise ValueError("Index arrays must contain integers")
            return v
        
        # List of integers - convert to array
        if isinstance(v, list):
            if not all(isinstance(x, int) for x in v):
                raise ValueError("Index lists must contain only integers")
            return np.array(v, dtype=int)
        
        # Single slice or int - wrap in tuple
        if isinstance(v, (slice, int)):
            return (v,)
        
        raise ValueError(f"Cannot convert {type(v)} to indices specification")
    
    def apply(self, array: np.ndarray) -> np.ndarray:
        """Apply indices to an array."""
        if self.value is None:
            return array.ravel()
        elif isinstance(self.value, tuple):
            return array[self.value].ravel()
        else:  # ndarray
            return array.ravel()[self.value]
    
    def model_dump(self, **kwargs) -> dict:
        """Serialize indices."""
        result = super().model_dump(**kwargs)
        
        if self.value is None:
            result['value'] = None
        elif isinstance(self.value, tuple):
            result['value'] = {
                'type': 'tuple',
                'value': serialize_index_tuple(self.value)
            }
        else:  # ndarray
            result['value'] = {
                'type': 'array',
                'value': self.value.tolist()
            }
        
        return result


class SrcIndicesSpec(IndicesSpec):
    """
    Specification for source indices in OpenMDAO connections and promotions.
    
    Used when connecting variables or promoting with specific indexing.
    Supports both shaped and flat indexing modes.
    
    Can be:
    - None: No indexing (1-to-1 connection)
    - tuple: Index tuple (from slicer or index_exp)
    - ndarray: Explicit integer indices (flat or shaped)
    - List of ints: Specific flat indices
    """
    
    src_shape: None | tuple[int, ...] = Field(
        default=None,
        description="Shape of source variable (for shaped indexing)"
    )
