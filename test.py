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
@withcomm()
def distributed_vector(comm):
  '''
  Create a uncompletely filled vector "v" with entries i^2 if i is odd and 0 otherwise
  and a vector "w" with all entries equal to i
  '''

  ndofs  = 20

  bounds = ( (ndofs-1) * numpy.arange( comm.nprocs+1, dtype=float ) / comm.nprocs ).astype( int )
  rowdofmap = map( numpy.arange, bounds[:-1], bounds[1:]+1 )
  rowmap = libmatrix.Map( comm, rowdofmap, ndofs )

  v   = libmatrix.VectorBuilder( rowmap )
  for idx in range(1,ndofs+1,2):
    v.add_global( idx=[numpy.array([idx])], data=[numpy.array([idx**2],dtype=float)] )
  v = v.complete()
  
  v_npy = numpy.arange(0,ndofs,dtype=float)**2
  v_npy[0::2] = 0. 

  numpy.testing.assert_almost_equal( v.toarray(), v_npy, err_msg='Vector "v" was not filled correctly' )

  #Test operations on v
  numpy.testing.assert_almost_equal( v.sum(), v_npy.sum(), err_msg='Vector sum incorrect' )
  numpy.testing.assert_almost_equal( v.norm(), numpy.linalg.norm(v_npy), err_msg='Vector norm incorrect' )
  numpy.testing.assert_almost_equal( v.copy().toarray(), v_npy, err_msg='Vector copy incorrect' )
  numpy.testing.assert_almost_equal( v.dot(v), (v_npy**2).sum(), err_msg='Vector dot-product with self incorrect' )
  numpy.testing.assert_almost_equal( (v+1).toarray(), v_npy+1, err_msg='Scalar addition not computed correctly' )
  numpy.testing.assert_almost_equal( (v-1).toarray(), v_npy-1, err_msg='Scalar subtraction not computed correctly' )
  numpy.testing.assert_almost_equal( (2.*v).toarray(), 2.*v_npy, err_msg='Scalar multiplication not computed correctly' )
  numpy.testing.assert_almost_equal( (v*2.).toarray(), 2.*v_npy, err_msg='Scalar multiplication not computed correctly' )
  numpy.testing.assert_almost_equal( (v/2.).toarray(), v_npy/2., err_msg='Scalar division not computed correctly' )
  numpy.testing.assert_almost_equal( (2./(v+1.)).toarray(), 2./(v_npy+1.), err_msg='Scalar division not computed correctly' )

  #Create the vector w based on the reversed rowdofmap
  w = libmatrix.VectorBuilder( rowmap )
  for idx in rowdofmap:
    w.add_global( idx=[idx], data=idx.astype(float)/numpy.bincount(numpy.concatenate(rowdofmap))[idx] )
  w = w.complete()

  w_npy = numpy.arange(ndofs)

  numpy.testing.assert_almost_equal( w.toarray(), w_npy, err_msg='Vector "w" was not filled correctly' )

  #Test binary operations on v and w
  numpy.testing.assert_almost_equal( v.dot(w), (w_npy*v_npy).sum(), err_msg='Innerproduct of "v" and "w" not computed correctly' )
  numpy.testing.assert_almost_equal( w.dot(v), (w_npy*v_npy).sum(), err_msg='Innerproduct of "w" and "v" not computed correctly' )
  numpy.testing.assert_almost_equal( (w-v).toarray(), w_npy-v_npy, err_msg='Substraction of "w" and "v" not computed correctly' )
  numpy.testing.assert_almost_equal( (w+v).toarray(), w_npy+v_npy, err_msg='Addition of "w" and "v" not computed correctly' )
  numpy.testing.assert_almost_equal( (w*v).toarray(), w_npy*v_npy, err_msg='Multiplication of "w" and "v" not computed correctly' )
  numpy.testing.assert_almost_equal( (w/(v+1.)).toarray(), w_npy/(v_npy+1.), err_msg='Division of "w" and "v" not computed correctly' )

@unittest
@withcomm(2)
def distributed_matrix(comm):
  '''
  Create the Rutishausser (default Toeppen) matrix from the Matlab gallery 
  '''

  ndofs  = 20

  coeffs = numpy.array([1.,-10,0.,10.,1.])
  A_npy  = sum([numpy.diag(coeffs[i]*numpy.ones(ndofs-abs(i-2)),k=-2+i) for i in range(5)])

  #Create overlapping row map
  bounds = ( (ndofs-1) * numpy.arange( comm.nprocs+1, dtype=float ) / comm.nprocs ).astype( int )
  rowdofmap = map( numpy.arange, bounds[:-1], bounds[1:]+1 )
  rowmap = libmatrix.Map( comm, rowdofmap, ndofs )

  #Create overlapping column map
  coldofmap = map( numpy.arange, numpy.maximum(bounds[:-1]-2,0), numpy.minimum(bounds[1:]+3,ndofs) )
  colmap = libmatrix.Map( comm, coldofmap, ownedmap=rowmap.ownedmap )

  #Initialize the matrix builder
  A = libmatrix.MatrixBuilder( (rowmap,colmap) )

  #Assemble the matrix
  block = coeffs[numpy.newaxis]
  for i in range(ndofs): 
    idx = numpy.arange(i-2,i+3)
    sel = numpy.greater(idx,-1)&numpy.less(idx,ndofs)
    A.add_global( idx=[[i],idx[sel]], data=block[:,sel] )
  A = A.complete()   

  numpy.testing.assert_almost_equal( A.toarray(), A_npy, err_msg= 'Matrix A was not filled correctly')

  #Matrix operations
  numpy.testing.assert_almost_equal( A.norm(), numpy.sqrt((A_npy**2).sum()), err_msg='Matrix norm not computed correctly' )

  #Matrix vector operations, x[i] = i
  x = libmatrix.VectorBuilder( colmap )
  for i in range(ndofs):
    x.add_global( idx=[numpy.array([i])], data=[numpy.array([i],dtype=float)]  )
  x = x.complete()  
  assert x.map is A.domainmap

  x_npy = numpy.arange(ndofs)
  numpy.testing.assert_almost_equal( x.toarray(), x_npy )

  #Matrix vector multiplication
  b = A.apply( x )
  assert b.map is A.rangemap

  b_npy = A_npy.dot( x_npy )
  numpy.testing.assert_almost_equal( b.toarray(), b_npy )

  #Matrix solve
  xsol = A.solve( rhs=b, precon=None, symmetric=False, tol=1e-10 )

  numpy.testing.assert_almost_equal( xsol.toarray(), x_npy )

@unittest
@withcomm(2)
def matrix_with_zero_blocks(comm):
  '''
  Create, constrain and solve the matrix [[0,I],[0,0]] 
  '''

  nfree = 7
  ncons = 3
  ndofs = nfree+ncons 

  A_npy = numpy.eye( ndofs, k=ncons )

  #Create overlapping row map
  bounds = ( (ndofs-1) * numpy.arange( comm.nprocs+1, dtype=float ) / comm.nprocs ).astype( int )
  rowdofmap = map( numpy.arange, bounds[:-1], bounds[1:]+1 )
  rowmap = libmatrix.Map( comm, rowdofmap, ndofs )

  #Create overlapping column map
  bounds = ( (ndofs-1) * numpy.arange( comm.nprocs+1, dtype=float ) / comm.nprocs ).astype( int )
  coldofmap = map( numpy.arange, bounds[:-1]+ncons, numpy.minimum(bounds[1:]+ncons+1,ndofs) )
  colmap = libmatrix.Map( comm, coldofmap, ownedmap=rowmap.ownedmap )

  #Initialize the matrix builder
  A = libmatrix.MatrixBuilder( (rowmap,colmap) )

  #Assemble the matrix
  for i in range(nfree): 
    A.add_global( idx=[[i],[i+ncons]], data=numpy.array([[1.]]) )
  A = A.complete()   

  numpy.testing.assert_almost_equal( A.toarray(), A_npy, err_msg= 'Matrix A was not filled correctly')

  #Vector x[i] = i
  x = libmatrix.VectorBuilder( A.rangemap )
  for i in range(ndofs):
    x.add_global( idx=[numpy.array([i])], data=[numpy.array([i],dtype=float)]  )
  x = x.complete()  

  x_npy = numpy.arange(ndofs,dtype=float)
  numpy.testing.assert_almost_equal( x.toarray(), x_npy )

  #Vector b[i] = ndofs-i-1
  b = libmatrix.VectorBuilder( A.domainmap )
  for i in range(ndofs):
    b.add_global( idx=[numpy.array([i])], data=[numpy.array([ndofs-i-1],dtype=float)]  )
  b = b.complete()  

  b_npy = numpy.arange(ndofs,0,-1,dtype=float)-1
  numpy.testing.assert_almost_equal( b.toarray(), b_npy )

  #Right constraints
  rcons = x.less( ncons-0.5 )
  rcons_npy = numpy.arange(ndofs,dtype=float)
  rcons_npy[ncons:] = numpy.nan
  numpy.testing.assert_almost_equal( rcons.toarray(), rcons_npy )

  #Left constraints
  lcons = b.less( ncons-0.5 )
  lcons_npy = numpy.arange(ndofs,0,-1,dtype=float)-1
  lcons_npy[:ndofs-ncons] = numpy.nan
  numpy.testing.assert_almost_equal( lcons.toarray(), lcons_npy )

  #Constrain the matrix
  #TODO
  #Does not work since the diagonal entries to be added are not
  #in the column map. Adding these to the above-defined column
  #map does not work, since in the exportAndFillComplete operation
  #these column indices will be lost.
  #Ac = A.constrained( lcons, rcons )

@unittest
@withcomm(2)
def solve_laplace(comm):
  '''
  Solve the 1D Laplace equation:
  d2udx2+1=0 on Omega=(0,1); u(0)=u(1)=1,
  using Bubnov-Galerkin linear FE. Exact solution:
  u = -(1/2)*x^2+(1/2)*x+1 
  '''

  nelems = 25
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
  v_npy =h* numpy.ones(ndofs)
  v_npy[0] /= 2
  v_npy[-1] /= 2
  numpy.testing.assert_almost_equal( v.toarray(), v_npy )

  #Vector operations
  vnorm_npy = numpy.sqrt((v_npy**2).sum())
  numpy.testing.assert_almost_equal( v.sum(), 1., err_msg='Vector sum' )
  numpy.testing.assert_almost_equal( v.norm(), vnorm_npy, err_msg='Vector norm' )
  numpy.testing.assert_almost_equal( v.dot(v), vnorm_npy**2, err_msg='Vector dot product' )

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

  #Maximum operation
  numpy.testing.assert_almost_equal( x.max(), x_npy.max() )

  #Compute the condition number estimate
  condest = Ac.condest( tol=1e-10, symmetric=True )
  numpy.testing.assert_almost_equal( condest, numpy.linalg.cond( Ac_npy, p=1 ) )

## END UNIT TESTS ##

unittest.exit()
