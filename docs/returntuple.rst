ReturnTuple Object
==================

Before we dig into the core aspects of the package, you will quickly notice
that many of the methods and functions defined here return a custom object
class. This return class is defined in :py:class:`biosppy.utils.ReturnTuple`.
The goal of this return class is to strengthen the semantic relationship
between a function's output variables, their names, and what is described in
the documentation. Consider the following function definition:


.. code:: python

    def compute(a, b):
        """Simultaneously compute the sum, subtraction, multiplication and
        division between two integers.

        Args:
            a (int): First input integer.
            b (int): Second input integer.

        Returns:
            (tuple): containing:
                sum (int): Sum (a + b).
                sub (int): Subtraction (a - b).
                mult (int): Multiplication (a * b).
                div (int): Integer division (a / b).

        """

        if b == 0:
            raise ValueError("Input 'b' cannot be zero.")

        v1 = a + b
        v2 = a - b
        v3 = a * b
        v4 = a / b

        return v1, v2, v3, v4

Note that Python doesn't actually support returning multiple objects. In this
case, the ``return`` statement packs the objects into a tuple.

.. code:: python

    >>> out = compute(4, 50)
    >>> type(out)
    <type 'tuple'>
    >>> print out
    (54, -46, 200, 0)

This is pretty straightforward, yet it shows one disadvantage of the native
Python return pattern: the semantics of the output elements (i.e. what each
variable actually represents) are only implicitly defined with the ordering
of the docstring. If there isn't a dosctring available (yikes!), the only way
to figure out the meaning of the output is by analyzing the code itself.

This is not necessarily a bad thing. One should always try to understand,
at least in broad terms, how any given function works. However, the initial
steps of the data analysis process encompass a lot of experimentation and
interactive exploration of the data. This is important in order to have an
initial sense of the quality of the data and what information we may be able to
extract. In this case, the user typically already knows what a function does,
but it is cumbersome to remember by heart the order of the outputs, without
having to constantly check out the documentation.

For instance, does the `numpy.histogram
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html>`__
function first return the edges or the values of the histogram? Maybe it's the
edges first, which correspond to the x axis. Oops, it's actually the other way
around...

In this case, it could be useful to have an explicit reference directly in the
return object to what each variable represents. Returning to the example above,
we would like to have something like:

.. code:: python

    >>> out = compute(4, 50)
    >>> print out
    (sum=54, sub=-46, mult=200, div=0)

This is exactly what :py:class:`biosppy.utils.ReturnTuple` accomplishes.
Rewriting the `compute` function to work with `ReturnTuple` is simple. Just
construct the return object with a tuple of strings with names for each output
variable:

.. code:: python

    from biosppy import utils

    def compute_new(a, b):
        """Simultaneously compute the sum, subtraction, multiplication and
        division between two integers.

        Args:
            a (int): First input integer.
            b (int): Second input integer.

        Returns:
            (ReturnTuple): containing:
                sum (int): Sum (a + b).
                sub (int): Subtraction (a - b).
                mult (int): Multiplication (a * b).
                div (int): Integer division (a / b).

        """

        if b == 0:
            raise ValueError("Input 'b' cannot be zero.")

        v1 = a + b
        v2 = a - b
        v3 = a * b
        v4 = a / b

        # build the return object
        output = utils.ReturnTuple((v1, v2, v3, v4), ('sum', 'sub', 'mult', 'div'))

        return output

The output now becomes:

.. code:: python

    >>> out = compute_new(4, 50)
    >>> print out
    ReturnTuple(sum=54, sub=-46, mult=200, div=0)

It allows to access a specific variable by key, like a dictionary:

.. code:: python

    >>> out['sum']
    54

And to list all the available keys:

.. code:: python

    >>> out.keys()
    ['sum', 'sub', 'mult', 'div']

It is also possible to convert the object to a more traditional dictionary,
specifically an `OrderedDict <https://docs.python.org/2/library/collections.html#collections.OrderedDict>`__:

.. code:: python

    >>> d = out.as_dict()
    >>> print d
    OrderedDict([('sum', 54), ('sub', -46), ('mult', 200), ('div', 0)])

Dictionary-like unpacking is supported:

.. code:: python

    >>> some_function(**out)

`ReturnTuple` is heavily inspired by `namedtuple <https://docs.python.org/2/library/collections.html#collections.namedtuple>`__,
but without the dynamic class generation at object creation. It is a subclass
of `tuple`, therefore it maintains compatibility with the native return pattern.
It is still possible to unpack the variables in the usual way:

.. code:: python

    >>> a, b, c, d = compute_new(4, 50)
    >>> print a, b, c, d
    54 -46 200 0

The behavior is slightly different when only one variable is returned. In this
case it is necessary to explicitly unpack a one-element tuple:

.. code:: python

    from biosppy import utils

    def foo():
        """Returns 'bar'."""

        out = 'bar'

        return utils.ReturnTuple((out, ), ('out', ))

.. code:: python

    >>> out, = foo()
    >>> print out
    'bar'
