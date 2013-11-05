import numpy, os
from mpi import MPI, InterComm


typemap = {
  'int32': (numpy.int32, MPI.INT),
  'int64': (numpy.int64, MPI.LONG),
  'float64': (numpy.float64, MPI.DOUBLE),
}


_info = dict( line.rstrip().split( ': ', 1 ) for line in os.popen( './libmatrix.mpi info' ) )

def bcast_token( func, names=_info.pop('token')[5:-1].split(', ') ):
  token = chr( names.index( func.func_name ) )
  def wrapped( self, *args, **kwargs ):
    assert self.isconnected(), 'connection is closed'
    self.bcast( token, TOKEN )
    return func( self, *args, **kwargs )
  return wrapped

LOCAL  = typemap[ _info.pop('local')  ]
GLOBAL = typemap[ _info.pop('global') ]
HANDLE = typemap[ _info.pop('handle') ]
SIZE   = typemap[ _info.pop('size')   ]
SCALAR = typemap[ _info.pop('scalar') ]
TOKEN  = numpy.character, MPI.CHAR

assert not _info
del _info


class LibMatrix( InterComm ):
  'interface to all libmatrix functions'

  def __init__( self, nprocs ):
    self.spawn( 'libmatrix.mpi', args=['eventloop'], maxprocs=nprocs )
    assert self.size == nprocs

  @bcast_token
  def new_map( self, globs ):
    lengths = map( len, globs )
    size = sum( lengths ) # TODO check meaning of size in map constructor
    self.bcast( size, SIZE )
    self.scatter( lengths, SIZE )
    self.scatterv( globs, GLOBAL )
    return self.gather_equal( HANDLE )

  @bcast_token
  def new_vector( self, map_handle ):
    self.bcast( map_handle, HANDLE )
    return self.gather_equal( HANDLE )

  @bcast_token
  def add_evec( self, handle, rank, idx, data ):
    n = len(idx)
    assert len(data) == n
    self.bcast( rank, SIZE )
    self.send( rank, handle, HANDLE )
    self.send( rank, n, SIZE )
    self.send( rank, idx, GLOBAL )
    self.send( rank, data, SCALAR )

  @bcast_token
  def get_vector( self, vec_handle, size, globs ):
    self.bcast( vec_handle, HANDLE )
    array = numpy.zeros( size ) # TODO fix length
    lengths = map( len, globs )
    local_arrays = self.gatherv( lengths, SCALAR )
    for idx, local_array in zip( globs, local_arrays ):
      array[idx] += local_array
    return array

  @bcast_token
  def new_matrix( self, graph_handle ):
    self.bcast( graph_handle, HANDLE )
    return self.gather_equal( HANDLE )

  @bcast_token
  def add_emat( self, handle, rank, rowidx, colidx, data ):
    data = numpy.asarray(data)
    shape = len(rowidx), len(colidx)
    assert data.shape == shape
    self.bcast( rank, SIZE )
    self.send( rank, handle, HANDLE )
    self.send( rank, shape, SIZE )
    print rowidx, colidx
    self.send( rank, rowidx, GLOBAL )
    self.send( rank, colidx, GLOBAL )
    self.send( rank, data.ravel(), SCALAR )

  @bcast_token
  def fill_complete( self, handle ):
    self.bcast( handle, HANDLE )

  @bcast_token
  def matrix_norm( self, handle ):
    self.bcast( handle, HANDLE )
    return self.gather_equal( SCALAR )

  @bcast_token
  def matvec_inplace( self, matrix_handle, out_handle, vec_handle ):
    self.bcast( [ matrix_handle, out_handle, vec_handle ], HANDLE )

  @bcast_token
  def new_graph( self, map_handle, rows ):
    self.bcast( map_handle, HANDLE )
    numcols = [ map( len, row ) for row in rows ]
    self.scatterv( numcols, SIZE )
    cols = [ numpy.concatenate( row ) for row in rows ]
    self.scatterv( cols, GLOBAL )
    return self.gather_equal( HANDLE )

  def __del__( self ):
    self.bcast( '\xff', TOKEN )
    InterComm.__del__( self )


#--- user facing objects ---


class Map( object ):

  def __init__( self, comm, globs ):
    self.comm = comm
    assert len(globs) == comm.size
    self.globs = [ numpy.asarray(glob,dtype=int) for glob in globs ]
    self.handle = comm.new_map( globs )


class Vector( object ):

  def __init__( self, comm, size, mp ):
    self.comm = comm
    self.size = size
    assert isinstance( mp, Map )
    self.mp = mp
    self.handle = comm.new_vector( mp.handle )

  def add( self, rank, idx, data ):
    self.comm.add_evec( self.handle, rank, idx, data )

  def toarray( self ):
    return self.comm.get_vector( self.handle, self.size, self.mp.globs )


class Matrix( object ):

  def __init__( self, comm, shape, graph ):
    self.comm = comm
    self.shape = shape
    assert isinstance( graph, Graph )
    self.graph = graph
    self.handle = comm.new_matrix( graph.handle )

  def add( self, rank, rowidx, colidx, data ):
    self.comm.add_emat( self.handle, rank, rowidx, colidx, data )

  def complete( self ):
    self.comm.fill_complete( self.handle )

  def norm( self ):
    return self.comm.matrix_norm( self.handle )

  def matvec( self, vec ):
    assert isinstance( vec, Vector )
    assert self.shape[1] == vec.size
    out = Vector( self.comm, self.shape[0], self.graph.mp )
    self.comm.matvec_inplace( self.handle, out.handle, vec.handle )
    return out


class Graph( object ):

  def __init__( self, comm, mp, rows ):
    self.comm = comm
    assert isinstance( mp, Map )
    self.mp = mp
    self.handle = comm.new_graph( mp.handle, rows )

