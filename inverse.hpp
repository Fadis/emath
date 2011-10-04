/***************************************************************************
 *   Copyright (C) 2009 by Naomasa Matsubayashi   *
 *   fadis@quaternion.sakura.ne.jp   *
 *                                                                         *
 *   All rights reserved.                                                  *
 *                                                                         *
 *   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: *
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. *
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. *
 *     * Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. *
 *                                                                         *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   *
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     *
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR *
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR *
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, *
 *   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,   *
 *   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    *
 *   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF *
 *   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS    *
 *   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
 ***************************************************************************/

#ifndef EMATH_INVERSE_HEADER
#define EMATH_INVERSE_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/memcpy.hpp>
#include <emath/lu.hpp>

namespace emath {
  template< unsigned int element_count, typename ElementType >
    EPP_DEVICE bool inverse( ElementType *_dest, ElementType *_source ) {
      unsigned int row_index[ element_count ];
      int index;
      for( index = 0; index != element_count; index++ )
        row_index[ index ] = index;

      int row;
      int col;
      for( row = 0; row != element_count; row++ )
        for( col = 0; col != element_count; col++ )
          _dest[ row * element_count + col ] = 0;

      if( !lu< element_count >( _source, row_index ) )
        return false;

      for( index = 0; index != element_count; index++ ) {
        for( row = 0; row != element_count; row++ ) {
          int pivot = row_index[ row ];
          ElementType sum = ( ( pivot == index ) ? 1 : 0 );
          for( col = 0; col != row; col++ )
            sum -= _source[ row * element_count + col ] * _dest[ col * element_count + index ];
          _dest[ row * element_count + index ] = sum;
        }
        for( row = element_count - 1; row != -1; row-- ) {
          ElementType sum = _dest[ row * element_count + index ];
          for( col = row + 1; col != element_count; col++ )
            sum -= _source[ row * element_count + col ] * _dest[ col * element_count + index ];
          _dest[ row * element_count + index ] = sum / _source[ row * element_count + row ];
        }
      }
      return true;
    }

  template< typename MatrixType >
    EPP_DEVICE bool inverse( MatrixType &_dest, const MatrixType &_source ) {
      typedef typename Traits< typename Traits< MatrixType >::ElementType >::ElementType ElementType;
      const unsigned int element_count = Traits< MatrixType >::element_count;
      unsigned int row_index[ element_count ];
      int index;
      for( index = 0; index != element_count; index++ )
        row_index[ index ] = index;

      MatrixType source_copy;
      int row;
      int col;
      for( row = 0; row != element_count; row++ )
        for( col = 0; col != element_count; col++ )
          get( source_copy, row, col ) = get( _source, row, col );

      if( !lu( source_copy, row_index ) )
        return false;

      for( index = 0; index != element_count; index++ ) {
        for( row = 0; row != element_count; row++ ) {
          int pivot = row_index[ row ];
          ElementType sum = ( ( pivot == index ) ? 1 : 0 );
          for( col = 0; col != row; col++ )
            sum -= get( source_copy, row, col ) * get( _dest, col, index );
          get( _dest, row, index ) = sum;
        }
        for( row = element_count - 1; row != -1; row-- ) {
          ElementType sum = get( _dest, row, index );
          for( col = row + 1; col != element_count; col++ )
            sum -= get( source_copy, row, col ) * get( _dest, col, index );
          get( _dest, row, index ) = sum / get( source_copy, row, row );
        }
      }
      return true;
    }

  template< typename MatrixType >
    EPP_DEVICE bool inverse( MatrixType &_dest ) {
      typedef typename Traits< typename Traits< MatrixType >::ElementType >::ElementType ElementType;
      const unsigned int element_count = Traits< MatrixType >::element_count;
      unsigned int row_index[ element_count ];
      int index;
      for( index = 0; index != element_count; index++ )
        row_index[ index ] = index;

      MatrixType source_copy;
      int row;
      int col;
      for( row = 0; row != element_count; row++ )
        for( col = 0; col != element_count; col++ )
          get( source_copy, row, col ) = get( _dest, row, col );

      if( !lu( source_copy, row_index ) )
        return false;

      for( index = 0; index != element_count; index++ ) {
        for( row = 0; row != element_count; row++ ) {
          int pivot = row_index[ row ];
          ElementType sum = ( ( pivot == index ) ? 1 : 0 );
          for( col = 0; col != row; col++ )
            sum -= get( source_copy, row, col ) * get( _dest, col, index );
          get( _dest, row, index ) = sum;
        }
        for( row = element_count - 1; row != -1; row-- ) {
          ElementType sum = get( _dest, row, index );
          for( col = row + 1; col != element_count; col++ )
            sum -= get( source_copy, row, col ) * get( _dest, col, index );
          get( _dest, row, index ) = sum / get( source_copy, row, row );
        }
      }
      return true;
    }

}
/*
#define EPP_INVERSE_MATRIX_FUNCTION( z, index, type ) \
  EPP_DEVICE bool inverse ( \
    EPP_MATRIX_TYPE( type, index ) &_matrix \
  ) { \
    EPP_SCALAR_TYPE( type ) source[ BOOST_PP_MUL( index, index ) ]; \
    EPP_SCALAR_TYPE( type ) dest[ BOOST_PP_MUL( index, index ) ]; \
    memcpy( source, _matrix ); \
    if( !inverse< index >( dest, source ) ) \
      return false; \
    memcpy( _matrix, dest ); \
    return true; \
  }

#define EPP_INVERSE_MATRIX_PROTOTYPE( z, index, type ) \
  EPP_DEVICE bool inverse ( \
    EPP_MATRIX_TYPE( type, index ) &_matrix \
  );


#define EPP_INVERSE_MATRIX( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_INVERSE_MATRIX_FUNCTION, \
    type \
  )

#define EPP_INVERSE_MATRIX_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_INVERSE_MATRIX_PROTOTYPE, \
    type \
  )
*/
namespace emath {
/*
#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_INVERSE_MATRIX, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_INVERSE_MATRIX_PROTOTYPES, EPP_UNUSED )
#endif
*/
}

#endif
