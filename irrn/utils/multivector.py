import torch as th
from typing import Union, Callable, Tuple, Any, Dict, Iterator, Type


class MultiVector(th.Tensor):
    """
    This class combines several tensors to be treated as a single vector
    """
    def __init__(self, elements: Tuple[th.Tensor]) -> None:
        """
        Initialize vector, given tensors data

        :param elements: several tensors to be combined into a single vector
        """
        ndims = []
        for t in elements:
            assert isinstance(t, th.Tensor)
            ndims.append(t.ndim)
        assert len(set(ndims)) == 1, \
            f'All elements of multivector should be of same dimension, received dimensions {ndims}.'
        self.elements = list(elements)
        self.num_elements = len(elements)

    @staticmethod
    def __new__(cls, elements: Tuple[th.Tensor], *args, **kwargs) -> Union['MultiVector', th.Tensor]:
        """
        Initialize vector, given tensors data

        :param elements: several tensors to be combined into a single vector
        """
        if len(elements) == 1:
            return elements[0]
        return super().__new__(cls, *args, **kwargs)

    def sum(self, reduce_parts: bool = True, **kwargs) -> Union['MultiVector', th.Tensor]:
        """
        This method performs sum operation for vector elements

        :param reduce_parts: if True, results over image and kernel parts are reduced via + and returned as th.Tensor;
                         if False, ImageKernel instance is returned with image and kernel parts as corresponding results
        :return: result of summation operation
        """
        ret = super().sum(**kwargs)
        if reduce_parts:
            return sum(ret)
        else:
            return ret

    def dim(self) -> int:
        """
        This method counts number of dimensions in vector parts, which are assumed to be equal through all subvectors.

        :return: number of dimensions of vector parts
        """
        return self.elements[0].dim()

    def all(self, reduce_parts: bool = True, **kwargs) -> Union[bool, Tuple[bool]]:
        """
        This method performs AND operation for vector elements

        :param reduce_parts: if True, results over multivector parts are reduced via AND and returned as th.Tensor;
                         if False, MultiVector instance is returned with all parts as corresponding results
        :return: result of AND operation
        """
        ret = super().all(**kwargs)
        if reduce_parts:
            return all(ret)
        else:
            return ret

    def __torch_function__(self, func: Callable, types: Tuple[Type], args: Tuple[Any] = (),
                           kwargs: Dict[str, Any] = None) -> Union['MultiVector', Tuple[Any]]:
        """
        This method overrides all PyTorch operations to be separately applied for each multivector parts
        """
        if kwargs is None:
            kwargs = {}
        elements_args = [[] for i in range(self.num_elements)]
        elements_kwargs = [{} for i in range(self.num_elements)]
        for arg in args:
            if isinstance(arg, MultiVector):
                assert arg.num_elements == self.num_elements
                for i in range(self.num_elements):
                    elements_args[i].append(arg.elements[i])
            else:
                for i in range(self.num_elements):
                    elements_args[i].append(arg)

        for key in kwargs:
            value = kwargs[key]
            if isinstance(value, MultiVector):
                assert value.num_elements == self.num_elements
                for i in range(self.num_elements):
                    elements_kwargs[i][key] = value.elements[i]
            else:
                for i in range(self.num_elements):
                    elements_kwargs[i][key] = value

        elements_output = []
        are_torch_tensors = []
        for i in range(self.num_elements):
            _args = tuple(elements_args[i])
            _kwargs = elements_kwargs[i]
            element_result = func(*_args, **_kwargs)
            elements_output.append(element_result)
            are_torch_tensors.append(isinstance(element_result, th.Tensor))
        elements_output = tuple(elements_output)
        if all(are_torch_tensors):
            return MultiVector(elements_output)
        else:
            return elements_output

    def __iter__(self) -> Iterator[th.Tensor]:
        """
        Method to correctly use tuple() and star operators. Returns a sequence of multivector elements.
        """
        return iter(self.elements)

    def __repr__(self):
        """
        Auxiliary method to correctly print data of ImageKernel instance
        """
        ret_str = 'Instance of MultiVector:\n'
        for num_elem in range(self.num_elements):
            ret_str += f'Element {num_elem}: {self.elements[num_elem]}\n'
        return ret_str

    @staticmethod
    def to_tuple(vector: Union[th.Tensor, 'MultiVector']) -> Tuple[th.Tensor]:
        """
        This method casts vector to tuple of tensors. If Tensor object is passed, returns tuple with single element,
        otherwise returns .elements attribute of MultiVector, which already is a tuple of tensors.

        :param vector: object to cast to tuple
        :return: tuple of tensors with elements from vector
        """
        if vector.__class__ == th.Tensor:
            return (vector, )
        else:
            assert vector.__class__ == MultiVector
            return tuple(vector.elements)
