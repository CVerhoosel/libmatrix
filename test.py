import sys, traceback, functools

class UnitTest( object ):
  ok = error = 0
  def __init__( self, width=60 ):
    self.tests = sys.argv[1:]
  def __call__( self, func ):
    if self.tests and func.func_name not in self.tests:
      return
    try:
      func()
    except AssertionError, e:
      print func.func_name, 'FAILED:', e
      self.error += 1
    except Exception:
      print func.func_name, 'ERROR'
      traceback.print_exc()
      self.error += 1
    else:
      print func.func_name, 'OK'
      self.ok += 1
    return func
  def exit( self ):
    print '%d tests successful, %d tests failed' % ( self.ok, self.error )
    raise SystemExit( self.error )

unittest = UnitTest()

## COMMUNICATOR UNITTEST ##
import multiprocessing
cpu_count = multiprocessing.cpu_count()

def withcomm( *args ):
  if not args:
    nprocs = 1
    while nprocs <= cpu_count:
      args += nprocs,
      nprocs *= 2
  def decorator( func ):  
    @functools.wraps( func )
    def wrapper():
      for nprocs in args:
        comm = libmatrix.LibMatrix( nprocs )
        func( comm )
    return wrapper
  return decorator
    
## BEGIN UNIT TESTS ##

#import __init__ as libmatrix
import libmatrix
import numpy

@unittest
@withcomm()
def create_communicators( comm ):
  assert isinstance( comm, libmatrix.LibMatrix )

@unittest
def create_overlapping_rowmap():
  comm = libmatrix.LibMatrix( nprocs=2 )
  rowdofmap = [numpy.array([0,1,2,3,4,8,9]),numpy.array([9,3,4,5,6,7])]
  rowmap = libmatrix.Map( comm, rowdofmap, 10 )

@unittest
@withcomm(2)
def solve_laplace(comm):
  '''
  Solve the 1D Laplace equation:
  d2udx2+1=0 on Omega=(0,1); u(0)=u(1)=1,
  using Bubnov-Galerkin linear FE. Exact solution:
  u = -(1/2)*x^2+(1/2)*x+1 
  '''

  nelems = 15
  ndofs  = nelems+1
  h      = 1./nelems

  bounds = ( (ndofs-1) * numpy.arange( comm.nprocs+1, dtype=float ) / comm.nprocs ).astype( int )
  rowdofmap = map( numpy.arange, bounds[:-1], bounds[1:]+1 )
  rowmap = libmatrix.Map( comm, rowdofmap, ndofs )

  #Building the vector
  v = libmatrix.VectorBuilder( rowmap )
  block = (h/2)*numpy.ones(2)
  for i in range(nelems):
    idx = numpy.arange(i,i+2)
    v.add_global( idx=[idx], data=block )
  v = v.complete()
  numpy.testing.assert_almost_equal( v.sum(), 1., err_msg='Integrated source term' )
  v_npy =h* numpy.ones(ndofs)
  v_npy[0] /= 2
  v_npy[-1] /= 2
  numpy.testing.assert_almost_equal( v.toarray(), v_npy )

  #Building the matrix
  block = (1./h)*numpy.array([[1,-1],[-1,1]])
  A = libmatrix.MatrixBuilder( rowmap )
  for i in range(nelems):
    idx = numpy.arange(i,i+2)
    A.add_global( idx=[idx,idx], data=block )
  A = A.complete()
  A_npy = (2./h)*numpy.diag(numpy.ones(ndofs))-(1./h)*numpy.diag(numpy.ones(nelems),k=1)-(1./h)*numpy.diag(numpy.ones(nelems),k=-1)
  A_npy[0,0] = 1./h
  A_npy[-1,-1] = 1./h
  numpy.testing.assert_almost_equal( A.toarray(), A_npy )

  #Creating constraints vector
  cons = v.less(0.75*h)
  cons *= (2./h)
  cons_npy = numpy.ones(ndofs)
  cons_npy[1:-1] = numpy.nan
  numpy.testing.assert_almost_equal( cons.toarray(), cons_npy )

  #Applying constraints
  Ac = A.constrained(cons)
  offdiag = (1./h)*numpy.ones(ndofs-1)
  offdiag[0] = 0.
  offdiag[-1] = 0.
  Ac_npy = (2./h)*numpy.diag(numpy.ones(ndofs))-numpy.diag(offdiag,k=1)-numpy.diag(offdiag,k=-1)
  Ac_npy[0,0] = 1.
  Ac_npy[-1,-1] = 1.
  numpy.testing.assert_almost_equal( Ac.toarray(), Ac_npy, err_msg='Applying constraints' )

  #Construct constrained rhs
  vc = cons | (v-A.apply( cons|0 ))
  vc_npy = h*numpy.ones(ndofs)
  vc_npy[0]  = 1.
  vc_npy[-1] = 1.
  vc_npy[1]  = h+1./h
  vc_npy[-2] = h+1./h
  numpy.testing.assert_almost_equal( vc.toarray(), vc_npy )
  
  #Solve the system using numpy
  x_npy = numpy.linalg.solve(Ac_npy,vc_npy)
  x_ex  = [-(1./2.)*c**2+(1./2.)*c+1. for c in numpy.linspace(0,1,ndofs)]
  numpy.testing.assert_almost_equal( x_npy, x_ex, err_msg='Verify against exact solution' )
  
  #Solve the system with constraints
  x = A.solve( rhs=v, precon=None, constrain=cons, symmetric=True, tol=1e-10 )
  numpy.testing.assert_almost_equal( x.toarray(), x_npy )

  #Solve the system with constraints and GMRES
  x = A.solve( rhs=v, precon=None, constrain=cons, name='GMRES', tol=1e-10 )
  numpy.testing.assert_almost_equal( x.toarray(), x_npy )

  #Solve the system with constraints and preconditioner
  x = A.solve( rhs=v, precon='ILUT', constrain=cons, symmetric=True, tol=1e-10 )
  numpy.testing.assert_almost_equal( x.toarray(), x_npy )

  #Solve the constrained matrix system
  x = Ac.solve( rhs=vc, precon=None, symmetric=True, tol=1e-10 )
  numpy.testing.assert_almost_equal( x.toarray(), x_npy )

## END UNIT TESTS ##

unittest.exit()
