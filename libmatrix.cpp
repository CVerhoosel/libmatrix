#include "typedefs.h"
#include "utils.h"

#include <mpi.h>

#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

#include <Ifpack2_Factory.hpp>

typedef int handle_t;
typedef uint8_t token_t;

#define ASSERT( cond ) if ( ! (cond) ) throw( "assertion failed: " #cond );

Teuchos::oblackholestream blackHole;

enum verbosity_t { ERROR, WARNING, INFO, DEBUG };

auto supportedPreconNames = Teuchos::tuple(
  std::string( "ILUT" ),
  std::string( "RILUK" ),
  std::string( "RELAXATION" ),
  std::string( "CHEBYSHEV" ),
  std::string( "DIAGONAL" ),
  std::string( "SCHWARZ" ),
  std::string( "KRYLOV" )
);

template <class T>
void print_all( const char name[], T iterable ) {
  std::cout << name;
  char sep[] = ": ";
  for ( auto const &item : iterable ) {
    std::cout << sep << item;
    sep[0] = ',';
  }
  std::cout << std::endl;
}

template <class T, class V>
inline bool contains( T iterable, V value ) {
  for ( auto const &item : iterable ) {
    if ( item == value ) {
      return true;
    }
  }
  return false;
}

typedef Kokkos::DefaultNode::DefaultNodeType node_t;
typedef Tpetra::Map<local_t,global_t,node_t> map_t;
typedef Tpetra::Operator<scalar_t,local_t,global_t,node_t> operator_t;
typedef Tpetra::MultiVector<scalar_t,local_t,global_t,node_t> multivector_t;
typedef Tpetra::RowMatrix<scalar_t,local_t,global_t,node_t> rowmatrix_t;
typedef Tpetra::CrsMatrix<scalar_t,local_t,global_t,node_t> crsmatrix_t;
typedef Tpetra::Vector<scalar_t,local_t,global_t,node_t> vector_t;
typedef Tpetra::CrsGraph<local_t,global_t,node_t> crsgraph_t;
typedef Tpetra::RowGraph<local_t,global_t,node_t> rowgraph_t;
typedef Belos::SolverFactory<scalar_t,multivector_t,operator_t> solverfactory_t;
typedef Teuchos::ScalarTraits<scalar_t>::magnitudeType magnitude_t;
typedef Tpetra::Export<local_t,global_t,node_t> export_t;
typedef Ifpack2::Preconditioner<scalar_t,local_t,global_t,node_t> precon_t;

inline Teuchos::Array<local_t> get_none_nan_entries( Teuchos::RCP<const vector_t>& nanvec )
{
  local_t i = 0;
  Teuchos::Array<local_t> items;
  for ( auto const &v : nanvec->getData() ) {
    if ( ! std::isnan( v ) ) {
      items.push_back( i );
    }
    i++;
  }
  return items;
}

class params_t : public Teuchos::Describable, public Teuchos::ParameterList {};

class linearproblem_t : public Teuchos::Describable, public Belos::LinearProblem<scalar_t,multivector_t,operator_t> {
public:
  linearproblem_t( const Teuchos::RCP<const operator_t> &A, const Teuchos::RCP<multivector_t> &X, const Teuchos::RCP<const multivector_t> &B) :
    Belos::LinearProblem<scalar_t,multivector_t,operator_t>( A, X, B ) {}
};


const global_t indexbase = 0;

const std::vector<std::string> funcnames = { FUNCNAMES };

class ConstrainedOperator : public operator_t {

public:

  ConstrainedOperator( Teuchos::RCP<const operator_t> op, Teuchos::Array<local_t> con_items )
    : op( op ), con_items( con_items ) {}

  const Teuchos::RCP<const map_t> &getDomainMap() const {
    return op->getDomainMap();
  }

  const Teuchos::RCP<const map_t> &getRangeMap() const {
    return op->getRangeMap();
  }

  void apply(
      const multivector_t &x,
      multivector_t &y,
      Teuchos::ETransp mode=Teuchos::NO_TRANS,
      scalar_t alpha=Teuchos::ScalarTraits<scalar_t>::one(),
      scalar_t beta=Teuchos::ScalarTraits<scalar_t>::zero() ) const {
    // y = alpha ( C A C + I - C ) x + beta y

    auto nvec = x.getNumVectors();
    multivector_t tmp1( x );
    for ( int ivec = 0; ivec < nvec; ivec++ ) {
      auto _tmp1 = tmp1.getDataNonConst( ivec );
      for ( const local_t i : con_items ) {
        _tmp1[i] = 0;
      }
    }
    multivector_t tmp2( y.getMap(), nvec, false );
    op->apply( tmp1, tmp2, mode );
    for ( int ivec = 0; ivec < nvec; ivec++ ) {
      auto _tmp2 = tmp2.getDataNonConst( ivec );
      auto _x = x.getData( ivec );
      for ( const local_t i : con_items ) {
        _tmp2[i] = _x[i];
      }
    }
    y.update( alpha, tmp2, beta );
  }

  bool hasTransposeApply() const {
    return true;
  }

private:
  
  const Teuchos::Array<local_t> con_items;
  const Teuchos::RCP<const operator_t> op;

};

class ChainOperator : public operator_t {

public:

  ChainOperator( Teuchos::RCP<const operator_t> op0, Teuchos::RCP<const operator_t> op1 )
    : op{ op0, op1 } { ASSERT( op0->getRangeMap() == op1->getDomainMap() ); }

  const Teuchos::RCP<const map_t> &getDomainMap() const {
    return op[0]->getDomainMap();
  }

  const Teuchos::RCP<const map_t> &getRangeMap() const {
    return op[1]->getRangeMap();
  }

  void apply(
      const multivector_t &x,
      multivector_t &y,
      Teuchos::ETransp mode=Teuchos::NO_TRANS,
      scalar_t alpha=Teuchos::ScalarTraits<scalar_t>::one(),
      scalar_t beta=Teuchos::ScalarTraits<scalar_t>::zero() ) const {
    // y = alpha op[1] op[0] x + beta y
    // => tmp = op[0] x
    //    y = alpha op[1] tmp + beta y
    multivector_t tmp( op[0]->getRangeMap(), y.getNumVectors() );
    const bool notrans = ( mode == Teuchos::NO_TRANS );
    op[notrans?0:1]->apply( x, tmp, mode );
    op[notrans?1:0]->apply( tmp, y, mode, alpha, beta );

  }

  bool hasTransposeApply() const {
    return true;
  }

private:
  
  const Teuchos::RCP<const operator_t> op[2];
};

class ObjectArray {

public:

  inline void set( handle_t handle, Teuchos::RCP<Teuchos::Describable> object, std::ostream &out ) {
    out << "set #" << handle << ": " << object->description() << std::endl;
    if ( handle == objects.size() ) {
      objects.push_back( object );
    }
    else {
      ASSERT( objects[handle].is_null() );
      objects[handle] = object;
    }
  }

  template <class T>
  inline Teuchos::RCP<T> get( handle_t handle, std::ostream &out ) {
    auto object = objects[handle];
    out << "get #" << handle << ": " << object->description() << std::endl;
    return Teuchos::rcp_dynamic_cast<T>( object, true );
  }

  inline void release( handle_t handle, std::ostream &out ) {
    out << "release #" << handle << std::endl;
    objects[handle] = Teuchos::RCP<Teuchos::Describable>();
  }

private:

  Teuchos::Array<Teuchos::RCP<Teuchos::Describable> > objects;

};

class Intercomm {

public:
  
  Intercomm( char *progname ) {
    int argc = 1;
    char **argv = &progname;
    MPI::Init( argc, argv ); 
    MPI::COMM_WORLD.Set_errhandler( MPI::ERRORS_THROW_EXCEPTIONS );
    intercomm = MPI::Comm::Get_parent();
    myrank = intercomm.Get_rank();
    nprocs = intercomm.Get_size();
  }

  ~Intercomm() {
    intercomm.Disconnect();
    MPI::Finalize(); 
  }

  template <class T>
  inline void bcast( T *data, const int n=1 ) {
    intercomm.Bcast( (void *)data, n * sizeof(T), MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void scatter( T *data, const int n=1 ) {
    intercomm.Scatter( NULL, 0, MPI::BYTE, (void *)data, n * sizeof(T), MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void scatterv( T *data, const int n=1 ) {
    intercomm.Scatterv( NULL, NULL, NULL, MPI::BYTE, (void *)data, n * sizeof(T), MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void recv( T *data, const int n=1, const int tag=0 ) {
    intercomm.Recv( (void *)data, n * sizeof(T), MPI::BYTE, tag, 0 );
  }
  
  template <class T>
  inline void gatherv( T *data, const int n=1 ) {
    intercomm.Gatherv( (void *)data, n * sizeof(T), MPI::BYTE, NULL, NULL, NULL, MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void gather( T *data, const int n=1 ) {
    intercomm.Gather( (void *)data, n * sizeof(T), MPI::BYTE, NULL, 0, MPI::BYTE, 0 );
  }

  void abort( int errorcode=1 ) {
    intercomm.Abort( errorcode );
  }

public:

  int myrank, nprocs;

private:

  MPI::Intercomm intercomm;

};

class LibMatrix : public Intercomm {

public:

  static int eventloop( char *progname ) {
    LibMatrix libmat( progname );
    typedef void (LibMatrix::*funcptr)();
    const funcptr ftable[] = { FUNCS };
    token_t c;
    for ( ;; ) {
      libmat.bcast( &c );
      if ( c >= funcnames.size() ) {
        libmat.out(DEBUG) << "quit" << std::endl;
        break;
      }
      libmat.out(DEBUG) << "enter " << funcnames[c] << std::endl;
      try {
        (libmat.*ftable[c])();
      }
      catch ( const char *s ) {
        libmat.out(ERROR) << "error in " << funcnames[c] << ": " << s << std::endl;
        libmat.abort();
        return 1;
      }
      libmat.out(DEBUG) << "leave " << funcnames[c] << std::endl;
    }
    Teuchos::TimeMonitor::summarize();
    return 0;
  }

private:

  LibMatrix( char *progname ) : Intercomm( progname ) {}

  void params_new() /* create new parameter list
     
      -> broadcast 1 HANDLE params_handle
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    auto params = Teuchos::rcp( new params_t );
  
    out(INFO) << "creating parameter list #" << handle.params << std::endl;
  
    objects.set( handle.params, params, out(DEBUG) );
  
  }
  
  void params_update() /* update parameter list from xml
     
      -> broadcast HANDLE params_handle 
      -> broadcast SIZE nchar
      -> broadcast CHAR xml[nchar]
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    size_t nchar;
    bcast( &nchar );
  
    std::string xml( nchar, 0 );
    bcast( const_cast<char*>(xml.data()), nchar );

    auto params = objects.get<params_t>( handle.params, out(DEBUG) );
    Teuchos::updateParametersFromXmlString( xml, params.ptr() );
  }
  
  void params_print() /* print the params list (c-sided)
     
      -> broadcast 1 HANDLE params_handle 
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    auto params = objects.get<params_t>( handle.params, out(DEBUG) );
    params->print( out(INFO) );
  }
  
  void release() /* release object
     
       -> broadcast HANDLE handle
  */{
  
    handle_t handle;
    bcast( &handle );
    objects.release( handle, out(DEBUG) );
  }
  
  void map_new() /* create new map
     
       -> broadcast HANDLE handle.map
       -> broadcast SIZE map_size
       -> scatter SIZE number_of_items[map_size]
       -> scatterv GLOBAL items[number_of_items]
  */{
  
    struct { handle_t map; } handle;
    bcast( &handle );
  
    size_t size, ndofs;
    bcast( &size );
    scatter( &ndofs );
  
    out(INFO) << "creating map #" << handle.map << " with " << ndofs << '/' << size << " items" << std::endl;
  
    Teuchos::Array<global_t> elementList( ndofs );
    scatterv( elementList.getRawPtr(), ndofs );
  
    auto node = Kokkos::DefaultNode::getDefaultNode();
    auto comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    auto map = Teuchos::rcp( new map_t( size, elementList, indexbase, comm, node ) );
  
    objects.set( handle.map, map, out(DEBUG) );
  }
  
  void vector_new() /* create new vector
     
       -> broadcast HANDLE handle.{vector,map}
  */{
  
    struct { handle_t vector, map; } handle;
    bcast( &handle );
  
    auto map = objects.get<const map_t>( handle.map, out(DEBUG) );
  
    out(INFO) << "creating vector #" << handle.vector << " from map #" << handle.map << std::endl;
  
    auto vector = Teuchos::rcp( new vector_t( map ) );
  
    objects.set( handle.vector, vector, out(DEBUG) );
  }
  
  void vector_add_block() /* add items to vector
     
       -> broadcast SIZE rank
     if rank == myrank
       -> recv HANDLE handle.vector
       -> recv SIZE number_of_items
       -> recv GLOBAL indices[number_of_items]
       -> recv SCALAR values[number_of_items]
     endif
  */{
  
    size_t rank;
    bcast( &rank );
  
    if ( rank != myrank ) {
      return;
    }
  
    struct { handle_t vector; } handle;
    recv( &handle );
  
    size_t nitems;
    recv( &nitems );
  
    out(INFO) << "ivec = " << handle.vector << ", nitems = " << nitems << std::endl;
  
    Teuchos::ArrayRCP<local_t> indices( nitems );
    Teuchos::ArrayRCP<scalar_t> data( nitems );
  
    recv( indices.getRawPtr(), nitems );
    recv( data.getRawPtr(), nitems );
  
    auto vec = objects.get<vector_t>( handle.vector, out(DEBUG) );
    auto value = data.begin();
    for ( auto const &idx : indices ) {
      out(INFO) << idx << " : " << *value << std::endl;
      vec->sumIntoLocalValue( idx, *value );
      value++;
    }
  
  }
  
  void vector_toarray() /* collect vector over the intercom
    
       -> broadcast HANDLE handle.vector
      <-  gather SIZE nitems
      <-  gatherv GLOBAL indices[nitems]
      <-  gatherv SCALAR values[nitems]
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );
  
    auto vec = objects.get<vector_t>( handle.vector, out(DEBUG) );
    auto map = vec->getMap();
    auto data = vec->getData();

    size_t nitems = map->getNodeNumElements();
    gather( &nitems );

    auto iitems = map->getNodeElementList();
    gatherv( iitems.getRawPtr(), iitems.size() );
  
    gatherv( data.get(), data.size() );
  }

  void vector_fill() /* fill vector with scalar value
    
       -> broadcast HANDLE handle.vector
       -> broadcast SCALAR value
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );

    scalar_t value;
    bcast( &value );
  
    auto vec = objects.get<vector_t>( handle.vector, out(DEBUG) );
    vec->putScalar( value );
  }
  
  void vector_truncateabove() /* replace values above threshold value
    
       -> broadcast HANDLE handle.self
       -> broadcast SCALAR threshold
       -> broadcast SCALAR replace
  */{
  
    struct { handle_t self; } handle;
    bcast( &handle );

    auto self = objects.get<vector_t>( handle.self, out(DEBUG) );

    scalar_t threshold;
    bcast( &threshold );

    scalar_t replace;
    bcast( &replace );

    for ( auto &self_i : self->getDataNonConst() ) {
      if ( self_i > threshold ){
        self_i = replace;
      }
    }

  }
  
  void vector_dot() /* innerproduct of two vectors
     
       -> broadcast HANDLE handle.{vector1,vector2}
      <-  gather SCALAR norm
  */{
  
    struct { handle_t vector1, vector2; } handle;
    bcast( &handle );
    auto vector1 = objects.get<vector_t>( handle.vector1, out(DEBUG) );
    auto vector2 = objects.get<vector_t>( handle.vector2, out(DEBUG) );
  
    ASSERT( vector1->getMap()==vector2->getMap() );

    scalar_t dot = vector1->dot( *vector2 );
  
    gather( &dot );
  }
  
  void vector_norm() /* compute L2-norm
     
       -> broadcast HANDLE handle.vector
      <-  gather SCALAR norm
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );
    auto vector = objects.get<vector_t>( handle.vector, out(DEBUG) );
  
    scalar_t norm = vector->norm2();
  
    gather( &norm );
  }

  void vector_abs() /* compute abs(vec)
     
       -> broadcast HANDLE handle.{vec,abs}
  */{

    struct { handle_t vec, abs; } handle;
    bcast( &handle );
  
    auto vec = objects.get<vector_t>( handle.vec, out(DEBUG) );
  
    out(INFO) << "creating vector #" << handle.abs << " from vector #" << handle.vec << std::endl;
  
    auto absvec = utils::vector::abs( vec );
  
    objects.set( handle.abs, absvec, out(DEBUG) );
  }
  
  void vector_sum() /* compute the sum of a vector
     
       -> broadcast HANDLE handle.vector
      <-  gather SCALAR sum
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );
    auto vector = objects.get<vector_t>( handle.vector, out(DEBUG) );

    scalar_t sum = 0.;
    for ( auto const &self_i : vector->getData() ) {
      sum+=self_i;
    }
  
    gather( &sum );
  }

  void vector_max() /* compute the max of a vector

       -> broadcast HANDLE handle.vector
      <-  gather SCALAR max
  */{

    struct { handle_t vector; } handle;
    bcast( &handle );
    auto vector = objects.get<vector_t>( handle.vector, out(DEBUG) );

    scalar_t max = NAN;
    for ( auto const &v : vector->getData() )
    {
      if( std::isnan(max) || v>max ) max=v;
    }

    gather( &max );
  }

  void vector_complete() /* export vector

       -> broadcast HANDLE handle.{vector,builder,exporter}
  */{

    struct { handle_t vector, builder, exporter; } handle;
    bcast( &handle );
    auto builder = objects.get<vector_t>( handle.builder, out(DEBUG) );
    auto exporter = objects.get<export_t>( handle.exporter, out(DEBUG) );
    auto map = exporter->getTargetMap();

    auto vector = Teuchos::rcp( new vector_t( map ) );
    vector->doExport( *builder, *exporter, Tpetra::ADD );

    objects.set( handle.vector, vector, out(DEBUG) );
  }

  void vector_or() /* logical OR vectors

       -> broadcast HANDLE handle.{self,other}
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    auto self = objects.get<vector_t>( handle.self, out(DEBUG) );
    auto other = objects.get<vector_t>( handle.other, out(DEBUG) );
    ASSERT( self->getMap() == other->getMap() );
    auto other_i = other->getData().begin();
    for ( auto &self_i : self->getDataNonConst() ) {
      if ( std::isnan( self_i ) ) {
        self_i = *other_i;
      }
      other_i++;
    }
  }

  void vector_and() /* logical AND vectors

       -> broadcast HANDLE handle.{self,other}
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    auto self = objects.get<vector_t>( handle.self, out(DEBUG) );
    auto other = objects.get<vector_t>( handle.other, out(DEBUG) );
    ASSERT( self->getMap() == other->getMap() );
    auto other_i = other->getData().begin();
    for ( auto &self_i : self->getDataNonConst() ) {
      if ( !std::isnan( self_i ) ) {
        self_i = *other_i;
      }
      other_i++;
    }
  }

  void vector_update() /* self += factor * other

       -> broadcast HANDLE handle.{self,other}
       -> broadcast SCALAR factor.{alpha,beta}
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    struct { scalar_t alpha, beta; } scalar;
    bcast( &scalar );
    auto self = objects.get<multivector_t>( handle.self, out(DEBUG) );
    auto other = objects.get<const multivector_t>( handle.other, out(DEBUG) );
    ASSERT( self->getMap() == other->getMap() );
    self->update( scalar.alpha, *other, scalar.beta );
  }

  void vector_imul() /* self *= other

       -> broadcast HANDLE handle.{self,other}
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    auto self = objects.get<vector_t>( handle.self, out(DEBUG) );
    auto other = objects.get<const vector_t>( handle.other, out(DEBUG) );
    ASSERT( self->getMap() == other->getMap() );
    auto other_i = other->getData().begin();
    for ( auto &self_i : self->getDataNonConst() ) {
      self_i *= *other_i;
      other_i++;
    }
  }

  void vector_idiv() /* self *= other

       -> broadcast HANDLE handle.{self,other}
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    auto self = objects.get<vector_t>( handle.self, out(DEBUG) );
    auto other = objects.get<const vector_t>( handle.other, out(DEBUG) );
    ASSERT( self->getMap() == other->getMap() );
    auto other_i = other->getData().begin();
    for ( auto &self_i : self->getDataNonConst() ) {
      self_i /= *other_i;
      other_i++;
    }
  }

  void vector_copy() /* logical OR vectors

       -> broadcast HANDLE handle.{copy,orig}
  */{

    struct { handle_t copy, orig; } handle;
    bcast( &handle );
    auto orig = objects.get<const vector_t>( handle.orig, out(DEBUG) );
    auto copy = Teuchos::rcp<vector_t>( new vector_t( *orig ) );
    objects.set( handle.copy, copy, out(DEBUG) );
  }
  
  void graph_new() /* create new graph
     
       -> broadcast HANDLE handle.{graph,rowmap,colmap,domainmap,rangemap}
       -> scatterv SIZE offsets[nrows+1]
       -> scatterv LOCAL columns[offsets[-1]]
  */{
  
    struct { handle_t graph, rowmap, colmap; } handle;
    bcast( &handle );
  
    auto rowmap = objects.get<const map_t>( handle.rowmap, out(DEBUG) );
    auto colmap = objects.get<const map_t>( handle.colmap, out(DEBUG) );
  
    size_t nrows = rowmap->getNodeNumElements();
  
    out(INFO) << "creating graph #" << handle.graph << " from rowmap #" << handle.rowmap << ", colmap #" << handle.colmap << " with " << nrows << " rows" << std::endl;
  
    Teuchos::ArrayRCP<size_t> offsets( nrows+1 );
    scatterv( offsets.getRawPtr(), nrows+1 );
  
    int nindices = offsets[nrows];
    Teuchos::ArrayRCP<local_t> indices( nindices );
    scatterv( indices.getRawPtr(), nindices );
  
    auto graph = Teuchos::rcp( new crsgraph_t( rowmap, colmap, offsets, indices ) );
    graph->fillComplete();
  
    objects.set( handle.graph, graph, out(DEBUG) );
  }
  
  void matrix_new_static() /* create new matrix
     
       -> broadcast HANDLE handle.{matrix,graph}
  */{
  
    struct { handle_t matrix, graph; } handle;
    bcast( &handle );
  
    auto graph = objects.get<const crsgraph_t>( handle.graph, out(DEBUG) );
  
    out(INFO) << "creating matrix #" << handle.matrix << " from graph #" << handle.graph << std::endl;
  
    auto matrix = Teuchos::rcp( new crsmatrix_t( graph ) );
  
    objects.set( handle.matrix, matrix, out(DEBUG) );
  }

  void matrix_new_dynamic() /* create new matrix

       -> broadcast HANDLE handle.{matrix,rowmap,colmap}
  */{

    struct { handle_t matrix, rowmap, colmap; } handle;
    bcast( &handle );

    out(INFO) << "creating matrix #" << handle.matrix << " from rowmap #" << handle.rowmap << " and colmap #" << handle.colmap << std::endl;

    auto rowmap = objects.get<const map_t>( handle.rowmap, out(DEBUG) );
    auto colmap = objects.get<const map_t>( handle.colmap, out(DEBUG) );

    auto matrix = Teuchos::rcp( new crsmatrix_t( rowmap, colmap, 0 ) );

    objects.set( handle.matrix, matrix, out(DEBUG) );
  }

  void matrix_add_block() /* add items to matrix
     
       -> broadcast SIZE rank
     if rank == myrank
       -> recv HANDLE handle.vector
       -> recv SIZE number_of_{rows,cols}
       -> recv GLOBAL indices[number_of_rows]
       -> recv GLOBAL indices[number_of_cols]
       -> recv SCALAR values[number_of_(rows*cols)]
     endif
  */{
  
    size_t rank;
    bcast( &rank );
  
    if ( rank != myrank ) {
      return;
    }
  
    struct { handle_t matrix; } handle;
    recv( &handle );
  
    size_t nitems[2];
    recv( nitems, 2 );
  
    out(INFO) << "imat = " << handle.matrix << ", nitems = " << nitems[0] << "," << nitems[1] << std::endl;
  
    Teuchos::ArrayRCP<local_t> rowidx( nitems[0] );
    Teuchos::ArrayRCP<local_t> colidx( nitems[1] );
    Teuchos::ArrayRCP<scalar_t> data( nitems[0] * nitems[1] );
  
    recv( rowidx.getRawPtr(), nitems[0] );
    recv( colidx.getRawPtr(), nitems[1] );
    recv( data.getRawPtr(), nitems[0] * nitems[1] );
  
    auto mat = objects.get<crsmatrix_t>( handle.matrix, out(DEBUG) );
    auto graph = mat->getCrsGraph();
  
    Teuchos::ArrayView<const local_t> current_icols;
    Teuchos::ArrayRCP<local_t> this_colidx( nitems[1] );
    Teuchos::ArrayRCP<scalar_t> this_data( nitems[1] );
    auto value = data.begin();
    for ( auto const &irow : rowidx ) {
      int nnew = 0, nold = 0;
      graph->getLocalRowView( irow, current_icols );
      for ( auto const &icol : colidx ) {
        if ( *value != 0 ) {
          int i = contains( current_icols, icol ) ? (nold++) : nitems[1] - (++nnew);
          this_colidx[i] = icol;
          this_data[i] = *value;
        }
        value++;
      }
      if ( nnew > 0 ) {
        out(INFO) << "inserting " << nnew << " new items in row " << irow << std::endl;
        mat->insertLocalValues( irow, this_colidx.view(nitems[1]-nnew,nnew), this_data.view(nitems[1]-nnew,nnew) );
      }
      if ( nold > 0 ) {
        out(INFO) << "adding " << nold << " existing items in row " << irow << std::endl;
        mat->sumIntoLocalValues( irow, this_colidx.view(0,nold), this_data.view(0,nold) );
      }
    }
  }

  void matrix_add() /* add matrix to matrix
     
       -> broadcast HANDLE handle.{self,other}
       -> broadcast SCALAR scalar.{alpha.beta}
  */{
  
    struct { handle_t sum, mat1, mat2; } handle;
    bcast( &handle );

    struct { scalar_t alpha, beta; } scalar;
    bcast( &scalar );

    auto mat1 = objects.get<const rowmatrix_t>( handle.mat1, out(DEBUG) );
    auto mat2 = objects.get<const rowmatrix_t>( handle.mat2, out(DEBUG) );
    auto sum = mat1->add( scalar.alpha, *mat2, scalar.beta );

    objects.set( handle.sum, sum, out(DEBUG) );
  }
  
  void matrix_complete() /* export matrix and fill-complete
     
       -> broadcast HANDLE handle.{matrix,builder,exporter,domainmap}
  */{
  
    struct { handle_t matrix, builder, exporter, domainmap; } handle;
    bcast( &handle );
  
    auto builder = objects.get<const crsmatrix_t>( handle.builder, out(DEBUG) );
    auto exporter = objects.get<const export_t>( handle.exporter, out(DEBUG) );
    auto domainmap = objects.get<const map_t>( handle.domainmap, out(DEBUG) );
    auto rangemap = exporter->getTargetMap();
  
    out(INFO) << "completing matrix #" << handle.builder << std::endl;
  
    auto matrix = Tpetra::exportAndFillCompleteCrsMatrix( builder, *exporter, domainmap, rangemap );
    // defaults to "ADD" combine mode (reverseMode=false in Tpetra_CrsMatrix_def.hpp)
  
    objects.set( handle.matrix, matrix, out(DEBUG) );
  }
  
  void matrix_norm() /* compute frobenius norm
     
       -> broadcast HANDLE handle.matrix
      <-  gather SCALAR norm
  */{
  
    struct { handle_t matrix; } handle;
    bcast( &handle );
    auto matrix = objects.get<crsmatrix_t>( handle.matrix, out(DEBUG) );
  
    scalar_t norm = matrix->getFrobeniusNorm();
  
    gather( &norm );
  }
  
  void operator_apply() /* matrix vector multiplication
     
       -> broadcast HANDLE handle.{matrix,out,vector}
  */{
  
    struct { handle_t matrix, rhs, lhs; } handle;
    bcast( &handle );
  
    auto matrix = objects.get<operator_t>( handle.matrix, out(DEBUG) );
    auto rhs = objects.get<vector_t>( handle.rhs, out(DEBUG) );
    auto lhs = objects.get<vector_t>( handle.lhs, out(DEBUG) );
  
    matrix->apply( *rhs, *lhs );
  }

  void matrix_condest() /* compute the operator's condition number

      -> broadcast HANDLE handle.operator
      -> gather SCALAR condest
  */{

    struct { handle_t oper, params, solvertype, precontype; } handle;
    bcast( &handle );

    auto B = objects.get<const crsmatrix_t>( handle.oper, out(DEBUG) );

    auto params = objects.get<Teuchos::ParameterList>( handle.params, out(DEBUG) );

    //Create the solver
    solverfactory_t factory;
    auto solverparams = Teuchos::sublist(params,"Solver Parameters");
    auto solvermgr = factory.create( factory.supportedSolverNames()[handle.solvertype], solverparams );

    //Create the preconditioner
    Teuchos::RCP<precon_t> precon;
    Ifpack2::Factory preconfactory;
    auto preconparams = Teuchos::sublist(params,"Preconditioner Parameters");
    if( handle.precontype!=-1 )
    {
      precon = preconfactory.create( supportedPreconNames[handle.precontype], B );
      precon->setParameters( *preconparams );
      precon->initialize();
      precon->compute();
    }

    ASSERT( B->getRangeMap()== B->getDomainMap() );

    //Compute the transpose of the operator
    auto BT = B;
    auto preconT = precon;
    if( !params->get("Symmetric",false) )
    {
      BT = utils::matrix::transpose( B );

      preconT = preconfactory.create( supportedPreconNames[handle.precontype], BT );
      preconT->setParameters( *preconparams );
      preconT->initialize();
      preconT->compute();
    }

    ASSERT( BT->getRangeMap()== B->getRangeMap() );
    ASSERT( BT->getRangeMap()== B->getRangeMap() );
    ASSERT( BT->getRowMap()  == B->getRangeMap() );

  /*+++++++++++++++++++++++
    + Matrix 1-norm       +
    +++++++++++++++++++++++*/

    auto norm1B = utils::matrix::norm1( B );

  /*+++++++++++++++++++++++
    + Hager's algorithm   +
    +++++++++++++++++++++++*/

    //Construct the initial vectors for
    auto x  = Teuchos::rcp( new vector_t( B->getRangeMap()  ) );
    auto y  = Teuchos::rcp( new vector_t( B->getDomainMap() ) );
    auto z  = Teuchos::rcp( new vector_t( B->getRangeMap()  ) );
    auto xi = Teuchos::rcp( new vector_t( B->getDomainMap() ) );

    x->putScalar( 1./x->getMap()->getGlobalNumElements() );

    auto linprob = Teuchos::rcp( new linearproblem_t( B , y, x  ) );
    auto adjprob = Teuchos::rcp( new linearproblem_t( BT, z, xi ) );

    if( handle.precontype != -1 )
    {
      linprob->setLeftPrec( precon  );
      adjprob->setLeftPrec( preconT );
    }

    if( params->get("Symmetric",false) )
    {
      linprob->setHermitian();
      adjprob->setHermitian();
    }

    global_t iiter;
    scalar_t gamma = NAN;

    for( iiter=0 ; iiter<=x->getMap()->getGlobalNumElements(); iiter++ )
    {
      y->putScalar( 0. );
      linprob->setProblem( y, x );

      solvermgr->setProblem( linprob );
      Belos::ReturnType result = solvermgr->solve();
      ASSERT( result == Belos::Converged );

      //Compute the sign vector
      auto xi_vals = xi->getDataNonConst();
      auto y_vals  = y->getData();

      ASSERT( xi_vals.size()==y_vals.size() );

      xi->putScalar(1.);
      for( local_t irow=0; irow<xi_vals.size() ; irow++ )
      {
        if( y_vals[irow] < 0. ) xi_vals[irow] = -1.;
      }
      z->putScalar( 0. );
      adjprob->setProblem( z, xi );

      solvermgr->setProblem( adjprob );
      result = solvermgr->solve();
      ASSERT( result == Belos::Converged );

      if( z->normInf()<=z->dot(*x) )
      {
        gamma = y->norm1();
        break;
      }

      scalar_t max    = NAN;
      local_t  argmax = -1;
      for( local_t irow=0 ; irow<z->getMap()->getNodeNumElements() ; irow++ )
      {
        if( std::isnan(max) || std::abs(z->getData()[irow])>max )
        {
           max=std::abs(z->getData()[irow]);
           argmax = irow;
        }
      }

      Teuchos::Array<scalar_t> allmax ( nprocs, 0. );
      Teuchos::Array<local_t> allargmax ( nprocs, -1 );
      Teuchos::gatherAll(*x->getMap()->getComm(),1,&max,nprocs,allmax.getRawPtr());
      Teuchos::gatherAll(*x->getMap()->getComm(),1,&argmax,nprocs,allargmax.getRawPtr());

      int maxrank = 0;
      max = allmax[0];
      for( int rank = 1 ; rank < nprocs ; rank++ )
      {
        if( allmax[rank]>max )
        {
          maxrank = rank;
          max     = allmax[rank];
        }
      }

      x->putScalar(0.);
      if( maxrank==myrank ) x->replaceLocalValue( allargmax[maxrank], 1. );
    }

    ASSERT( !std::isnan(gamma) );

    out(INFO) << "Hager's algorithm converged in " << iiter << " iterations." << std::endl;

    scalar_t condest = norm1B*gamma;

    gather( &condest );
  }

  void matrix_toarray() /* matrix vector multiplication
     
       -> broadcast HANDLE handle.matrix
      <-  gather SIZE ncols
      <-  gather SIZE nrows
      <-  gatherv GLOBAL irows[nrows]
      <-  gatherv SIZE nentries_per_row[nrows]
      <-  gatherv GLOBAL icols[nentries]
      <-  gatherv SCALAR values[nentries]
  */{
  
    struct { handle_t matrix; } handle;
    bcast( &handle );
  
    auto matrix = objects.get<crsmatrix_t>( handle.matrix, out(DEBUG) );
    auto domainmap = matrix->getDomainMap();
    auto rowmap = matrix->getRowMap();
    auto colmap = matrix->getColMap();

    size_t ncols = domainmap->getNodeNumElements();
    gather( &ncols );

    size_t nrows = rowmap->getNodeNumElements();
    gather( &nrows );

    auto irows = rowmap->getNodeElementList();
    gatherv( irows.getRawPtr(), irows.size() );

    Teuchos::ArrayRCP<size_t> nentries_per_row( nrows );
    local_t irow = 0;
    int nentries = 0;
    for ( auto &entry : nentries_per_row ) {
      entry = matrix->getNumEntriesInLocalRow( irow );
      nentries += entry;
      irow++;
    }
    gatherv( nentries_per_row.get(), nentries_per_row.size() );

    Teuchos::ArrayRCP<global_t> all_indices( nentries );
    Teuchos::ArrayRCP<scalar_t> all_values( nentries );
    Teuchos::ArrayView<const local_t> indices;
    Teuchos::ArrayView<const scalar_t> values;
    int offset = 0;
    for ( local_t irow = 0; irow < nrows; irow++ ) {
      matrix->getLocalRowView( irow, indices, values );
      if ( values.size() ) {
        all_values.view( offset, values.size() ).assign( values );
        for ( auto idx : indices ) {
          all_indices[offset] = colmap->getGlobalElement( idx );
          offset++;
        }
      }
    }
    gatherv( all_indices.get(), all_indices.size() );
    gatherv( all_values.get(), all_values.size() );
  }

  void operator_constrained() /* matrix vector multiplication
     
       -> broadcast HANDLE handle.{conmat,matrix,vector}
  */{
  
    struct { handle_t conmat, matrix, vector; } handle;
    bcast( &handle );
  
    auto matrix = objects.get<const operator_t>( handle.matrix, out(DEBUG) );
    auto vector = objects.get<const vector_t>( handle.vector, out(DEBUG) );
    Teuchos::Array<local_t> con_items;
    local_t ldof = 0;
    for ( auto const &v : vector->getData() ) {
      if ( ! std::isnan( v ) ) {
        con_items.push_back( ldof );
      }
      ldof++;
    }

    auto comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    auto conmat = Teuchos::rcp( new ConstrainedOperator( matrix, con_items ) );

    objects.set( handle.conmat, conmat, out(DEBUG) );
  }

  void matrix_constrained() /* applying constraints to matrix
     
       -> broadcast HANDLE handle.{constrained,matrix,lcons,rcons}
  */{
  
    struct { handle_t constrained, matrix, lcons, rcons; } handle;
    bcast( &handle );

    //Get the original matrix and constraints vector
    auto matrix = objects.get<const crsmatrix_t>( handle.matrix, out(DEBUG) );
    auto lcons  = objects.get<const vector_t>( handle.lcons, out(DEBUG) );
    auto rcons  = objects.get<const vector_t>( handle.rcons, out(DEBUG) );

    ASSERT( matrix->isFillComplete() );
    ASSERT( matrix->getRangeMap()==matrix->getRowMap() );
    ASSERT( matrix->getRangeMap()==matrix->getDomainMap() );
    ASSERT( lcons->getMap()==matrix->getRangeMap() );
    ASSERT( rcons->getMap()==matrix->getDomainMap() );

    //Get the constrained indices
    auto lcon_items = get_none_nan_entries( lcons );
    auto rcon_items = get_none_nan_entries( rcons );

    //Create the to-be-filled constrained matrix
    auto builder = Teuchos::rcp( new crsmatrix_t( matrix->getRowMap(), matrix->getColMap(), 0 ) );

    Teuchos::ArrayView<const local_t> icols;
    Teuchos::ArrayView<const scalar_t> ivals;

    Teuchos::Array<local_t> diagcol ( 1 );
    Teuchos::Array<scalar_t> diagval ( 1, 1. );

    Teuchos::Array<local_t>  cols;
    Teuchos::Array<scalar_t> vals;
    local_t irow, icol, idom;
    for( irow = 0 ; irow < matrix->getNodeNumRows() ; irow++ )
    {
      matrix->getLocalRowView ( irow,  icols, ivals );

      if( !contains(lcon_items,irow) )
      {
        cols.reserve( icols.size() );
        vals.reserve( ivals.size() );
        for( icol = 0 ; icol < icols.size() ; icol++ )
        {
           idom = rcons->getMap()->getLocalElement(matrix->getColMap()->getGlobalElement( icols[icol] ));

           if( !contains(rcon_items, idom ) )
           {
             cols.push_back( icols[icol] );
             vals.push_back( ivals[icol] );
           }
        }
        builder->insertLocalValues ( irow,  cols(), vals() );
        cols.clear();
        vals.clear();
      }
      else
      {
        diagcol[0] = matrix->getColMap()->getLocalElement(matrix->getRowMap()->getGlobalElement(irow));
        builder->insertLocalValues ( irow, diagcol(), diagval() );
      }
    }


    //Fill complete
    builder->fillComplete( matrix->getDomainMap(), matrix->getRangeMap() );

    objects.set( handle.constrained, builder, out(DEBUG) );
  }
  
  void vector_nan_from_supp() /* set vector items to nan for non suppored rows
     
       -> broadcast HANDLE handle.{vector,matrix}
  */{
  
    struct { handle_t vector, matrix; } handle;
    bcast( &handle );
  
    auto vector = objects.get<vector_t>( handle.vector, out(DEBUG) );
    auto matrix = objects.get<const crsmatrix_t>( handle.matrix, out(DEBUG) );
    auto graph = matrix->getGraph();
    local_t irow = 0;
    for ( auto &vector_i : vector->getDataNonConst() ) {
      if ( graph->getNumEntriesInLocalRow( irow ) == 0 ) {
        vector_i = NAN;
      }
      irow++;
    }
  }
  
  void precon_new() /* create new preconditioner
     
       -> broadcast HANDLE handle.{precon,matrix,precontype,preconparams}
  */{
  
    struct { handle_t precon, matrix, precontype, preconparams; } handle;
    bcast( &handle );
  
    auto matrix = objects.get<const crsmatrix_t>( handle.matrix, out(DEBUG) );
    auto preconparams = objects.get<const params_t>( handle.preconparams, out(DEBUG) );
   
    Ifpack2::Factory factory;
    auto precon = factory.create( supportedPreconNames[handle.precontype], matrix );
  
    precon->setParameters( *preconparams );
    precon->initialize();
    precon->compute();
  
    objects.set( handle.precon, precon, out(DEBUG) );
  }

  void export_new() /* create new exporter
     
       -> broadcast HANDLE handle.{exporter,srcmap,dstmap}
  */{
  
    struct { handle_t exporter, srcmap, dstmap; } handle;
    bcast( &handle );
  
    auto srcmap = objects.get<const map_t>( handle.srcmap, out(DEBUG) );
    auto dstmap = objects.get<const map_t>( handle.dstmap, out(DEBUG) );
  
    auto exporter = Teuchos::rcp( new export_t( srcmap, dstmap ) );
  
    objects.set( handle.exporter, exporter, out(DEBUG) );
  }

  void linearproblem_new() /* create new linear problem
     
       -> broadcast HANDLE handle.{linprob,matrix,lhs0}
  */{
  
    struct { handle_t linprob, matrix, lhs, rhs; } handle;
    bcast( &handle );
  
    auto matrix = objects.get<const operator_t>( handle.matrix, out(DEBUG) );
    auto lhs = objects.get<vector_t>( handle.lhs, out(DEBUG) );
    auto rhs = objects.get<const vector_t>( handle.rhs, out(DEBUG) );

    ASSERT( matrix->getDomainMap()==lhs->getMap() );
    ASSERT( matrix->getRangeMap()==rhs->getMap() );

    auto linprob = Teuchos::rcp( new linearproblem_t( matrix, lhs, rhs ) );
  
    objects.set( handle.linprob, linprob, out(DEBUG) );
  }

  void linearproblem_set_hermitian() /* tell that operator is hermitian
     
       -> broadcast HANDLE handle.linprob
  */{
  
    struct { handle_t linprob; } handle;
    bcast( &handle );
  
    auto linprob = objects.get<linearproblem_t>( handle.linprob, out(DEBUG) );
    linprob->setHermitian();
  }

  void linearproblem_set_precon() /* set left/right preconditioner
     
       -> broadcast HANDLE handle.{linprob,prec}
       -> broadcast BOOL side
  */{
  
    struct { handle_t linprob, precon; } handle;
    bcast( &handle );

    bool_t right;
    bcast( &right );

    auto linprob = objects.get<linearproblem_t>( handle.linprob, out(DEBUG) );
    auto precon = objects.get<const operator_t>( handle.precon, out(DEBUG) );

    if ( right ) {
      linprob->setRightPrec( precon );
    }
    else {
      linprob->setLeftPrec( precon );
    }
  }

  void linearproblem_solve() /* solve system
     
       -> broadcast HANDLE handle.{linprob,solverparams,solvertype}
  */{
  
    struct { handle_t linprob, solverparams, solvertype; } handle;
    bcast( &handle );
  
    auto linprob = objects.get<linearproblem_t>( handle.linprob, out(DEBUG) );
    auto solverparams = objects.get<params_t>( handle.solverparams, out(DEBUG) );

    solverfactory_t factory;
    auto solvermgr = factory.create( factory.supportedSolverNames()[handle.solvertype], solverparams );
  
    // called on the linear problem, before they can solve it.
    linprob->setProblem();
  
    // Tell the solver what problem you want to solve.
    solvermgr->setProblem( linprob );

    // Attempt to solve the linear system.  result == Belos::Converged
    // means that it was solved to the desired tolerance.  This call
    // overwrites X with the computed approximate solution.
    Belos::ReturnType result = solvermgr->solve();
    ASSERT( result == Belos::Converged );

    const int numIters = solvermgr->getNumIters();
    out(INFO) << "solver converged in " << numIters << " iterations" << std::endl;
  }

  void set_verbosity() /* set minimum displayed log level

      -> NUMBER level
  */{
    bcast( &max_verbosity_level );
  }

  inline std::ostream& out( verbosity_t level ) {
    return level <= max_verbosity_level ? std::cout << '[' << (myrank+1) << '/' << nprocs << "] " : blackHole;
  }

private:

  int max_verbosity_level = WARNING;
  ObjectArray objects;

};

int main( int argc, char *argv[] ) {
  if ( argc == 2 && std::strcmp( argv[1], "eventloop" ) == 0 ) {
    return LibMatrix::eventloop( argv[0] );
  }
  print_all( "functions", funcnames );
  print_all( "solvers", solverfactory_t().supportedSolverNames() );
  print_all( "precons", supportedPreconNames );
  std::cout << "token_t: uint" << (sizeof(token_t) << 3) << std::endl;
  std::cout << "local_t: int" << (sizeof(local_t) << 3) << std::endl;
  std::cout << "global_t: int" << (sizeof(global_t) << 3) << std::endl;
  std::cout << "size_t: uint" << (sizeof(size_t) << 3) << std::endl;
  std::cout << "handle_t: int" << (sizeof(handle_t) << 3) << std::endl;
  std::cout << "number_t: int" << (sizeof(number_t) << 3) << std::endl;
  std::cout << "bool_t: uint" << (sizeof(bool_t) << 3) << std::endl;
  std::cout << "scalar_t: float" << (sizeof(scalar_t) << 3) << std::endl;
  return 1;
}


// vim:foldmethod=syntax
