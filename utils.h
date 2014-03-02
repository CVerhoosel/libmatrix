#ifndef LIBMATRIX_UTILS_H
#define LIBMATRIX_UTILS_H

#include "typedefs.h"

#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsMatrix.hpp>

typedef Kokkos::DefaultNode::DefaultNodeType node_t;
typedef Tpetra::CrsMatrix<scalar_t,local_t,global_t,node_t> crsmatrix_t;
typedef Tpetra::Vector<scalar_t,local_t,global_t,node_t> vector_t;

namespace utils
{
  namespace vector
  {
    Teuchos::RCP<vector_t> abs ( Teuchos::RCP<vector_t> );

    //local_t argmax ( )
  }

  namespace matrix
  {
    Teuchos::RCP<crsmatrix_t> transpose ( const Teuchos::RCP<const crsmatrix_t>  );

    scalar_t norminf ( const Teuchos::RCP<const crsmatrix_t> );

    scalar_t norm1 ( const Teuchos::RCP<const crsmatrix_t> );
  }
}

#endif
