class UnitTest( object ):
  ok = error = 0
  def __init__( self, width=60 ):
    self.width = width
    print '-' * self.width
  def __call__( self, func, *args ):
    s = '%s ... ' % func.func_name
    try:
      func( *args )
    except Exception, e:
      s += repr(e)
      print s + 'ERROR'.rjust(self.width-len(s))
      self.error += 1
    else:
      print s + 'OK'.rjust(self.width-len(s))
      self.ok += 1
    return func
  def exit( self ):
    print '-' * self.width
    print '%d tests successful, %d tests failed' % ( self.ok, self.error )
    raise SystemExit( self.error )

unittest = UnitTest()

## COMMUNICATOR UNITTEST ##
import multiprocessing
cpu_count = multiprocessing.cpu_count()

def unittest_comm ( func ):
  nprocs = 1
  while nprocs <= cpu_count:
    comm = libmatrix.LibMatrix( nprocs )
    unittest( func, comm )
    nprocs *= 2

## BEGIN UNIT TESTS ##

#import __init__ as libmatrix
import libmatrix
import numpy

@unittest_comm
def create_communicators ( comm ):
  assert isinstance( comm, libmatrix.LibMatrix )

@unittest
def create_overlapping_rowmap():
  comm = libmatrix.LibMatrix( nprocs=2 )
  rowdofmap = [numpy.array([0,1,2,3,4,8,9]),numpy.array([9,3,4,5,6,7])]
  rowmap = libmatrix.Map( comm, rowdofmap, 10 )

## END UNIT TESTS ##

unittest.exit()
