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
tuples, in that priority.

More generally, this is a fun project that is teaching me some of the nooks and
crannies of the respective languages.

## Usage

    from cljpy import core
    
    core.assoc({'one': 1}, 'two', 2)
    
    => {'one': 1, 'two': 2}

## Testing

    py.test tests/

## License

Copyright (C) 2011 Jim Blomo

Distributed under the Eclipse Public License, the same as Clojure.
