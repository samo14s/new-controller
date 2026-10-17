"""Path setup shared by the verification scripts in this folder."""
import os
import sys

# single-threaded BLAS: the small matrices here run 100x slower with thread
# oversubscription on a shared machine.  Must run before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def out(name):
    return os.path.join(HERE, name)
