#include "utils.h"

#include <Tpetra_RowMatrixTransposer.hpp>

/*
Vector utilities
*/

Teuchos::RCP<vector_t> utils::vector::abs ( Teuchos::RCP<vector_t> vector )
{
  auto absvector = Teuchos::rcp(new vector_t( vector->getMap() ));
  for( local_t irow=0 ; irow<absvector->getMap()->getNodeNumElements() ; irow++ )
  {
    absvector->getDataNonConst()[irow] = std::abs(vector->getData()[irow]);
  }
  return absvector;
}


/*
Matrix utilities
*/

Teuchos::RCP<crsmatrix_t> utils::matrix::transpose ( const Teuchos::RCP<const crsmatrix_t> matrix )
{
  Tpetra::RowMatrixTransposer<scalar_t,local_t,global_t,node_t> transposer ( matrix );
  return transposer.createTranspose();
}


scalar_t utils::matrix::norminf ( const Teuchos::RCP<const crsmatrix_t> matrix )
{
  scalar_t norm = 0.;
  scalar_t rowsum;
  Teuchos::ArrayView<const local_t> indices;
  Teuchos::ArrayView<const scalar_t> values;

  for( local_t irow=0 ; irow<matrix->getRowMap()->getNodeNumElements() ; irow++ )
  {
    matrix->getLocalRowView(irow, indices, values);
    rowsum = 0.;
    for( auto &v : values ) rowsum += std::abs(v);
    if( rowsum > norm ) norm=rowsum;
  }

  if (matrix->isDistributed())
  {
    scalar_t lnorm = norm;
    Teuchos::reduceAll(*matrix->getRowMap()->getComm(),Teuchos::REDUCE_MAX,lnorm,Teuchos::outArg(norm));
  }

  return norm;
}


scalar_t utils::matrix::norm1 ( const Teuchos::RCP<const crsmatrix_t> matrix )
{
  return utils::matrix::norminf( utils::matrix::transpose(matrix) );
}
