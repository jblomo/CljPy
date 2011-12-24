# CljPy - Clojure function in Python

Ever started work on a Python project but missed the functional tools that come
with Clojure?  Just `import cljpy` and the functions you love will be at your
fingertips again!

## Goals

Achieve feature parity to Clojure concepts with idiomatic Python.  Where
idiomatic differences between the languages arise, side with Clojure.  The most
prominent example is immutable types.  This module will use the Python copy
module to avoid destructive updates of arguments.  For iterable return values,
it tries to return generators, the type of the original function argument, or
tuples, in that priority.  Macros are implemented either as higher order
functions (eg. condp) or decorators (eg. defmulti/defmethod).

Still undecided how to handle:

- Clojure style references
- namespace manipulation

More generally, this is a fun project that is teaching me some of the nooks and
crannies of the respective languages.

## Usage

    import operator
    from cljpy import core
    
    core.merge_with(operator.add, {'one': 1, 'seven': 3}, {'two': 2, 'seven': 4})
    
    => {'one': 1, 'two': 2, 'seven': 7}

## Testing

    py.test tests/

## License

Copyright (C) 2011 Jim Blomo

Distributed under the Eclipse Public License, the same as Clojure.
