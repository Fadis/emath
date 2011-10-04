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

#ifndef EMATH_LU_HEADER
#define EMATH_LU_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

namespace emath {
  template< unsigned int element_count, typename ElementType >
    EPP_DEVICE bool lu( ElementType *_lu, unsigned int *_index ) {
      ElementType weight[ element_count ];
      int row;
      for( row = 0; row != element_count; row++ ) {
        ElementType max_in_row = 0.0;
        int col;
        for( col = 0; col != element_count; col++ ) {
          ElementType distance = _lu[ row * element_count + col ];
          if( distance < 0.0 )
            distance = -distance;
          if( distance > max_in_row )
            max_in_row = distance;
        }
        if( max_in_row == 0.0 )
          return false;
        weight[ row ] = 1.0 / max_in_row;
      }
      int col;
      for( col = 0; col != element_count; col++ ) {
        for( row = 0; row != col; row++ ) {
          ElementType sum = _lu[ row * element_count + col ];
          int index;
          for( index = 0; index != row; index++ )
            sum -= _lu[ row * element_count + index ] * _lu[ index * element_count + col ];
          _lu[ row * element_count + col ] = sum;
        }
        int max_rated_row = 0.0;
        ElementType max_rate = 0.0;
        for( row = col; row != element_count; row++ ) {
          ElementType sum = _lu[ row * element_count + col ];
          int index;
          for( index = 0; index != col; index++ )
            sum -= _lu[ row * element_count + index ] * _lu[ index * element_count + col ];
          _lu[ row * element_count + col ] = sum;
          if( sum < 0.0 )
            sum = -sum;
          ElementType rate = weight[ row ] * sum;
          if( rate >= max_rate ){
            max_rate = rate;
            max_rated_row = row;
          }
        }
        row = col;
        if( row != max_rated_row ) {
          int index;
          for( index = 0; index != element_count; index++ ) {
            {
              ElementType temp = _lu[ max_rated_row * element_count + index ];
              _lu[ max_rated_row * element_count + index ] = _lu[ row * element_count + index ];
              _lu[ row * element_count + index ] = temp;
            }
          }
          {
            unsigned int temp = _index[ row ];
            _index[ row ] = _index[ max_rated_row ];
            _index[ max_rated_row ] = temp;
          }
          weight[ max_rated_row ] = weight[ row ];
        }
        if( _lu[ row * element_count + row ] == 0.0 )
          return false;
        if( row != element_count - 1 ) {
          ElementType temp = 1.0 / _lu[ row * element_count + row ];
          for( row = col + 1; row < element_count; row++ )
            _lu[ row * element_count + col ] *= temp;
        }
      }
      return true;
    }

  template< typename MatrixType >
    EPP_DEVICE bool lu( MatrixType &_lu, unsigned int *_index ) {
      typedef typename Traits< typename Traits< MatrixType >::ElementType >::ElementType ElementType;
      const unsigned int element_count = Traits< MatrixType >::element_count;
      ElementType weight[ element_count ];
      int row;
      for( row = 0; row != element_count; row++ ) {
        ElementType max_in_row = 0.0;
        int col;
        for( col = 0; col != element_count; col++ ) {
          ElementType distance = get( _lu, row, col );
          if( distance < 0.0 )
            distance = -distance;
          if( distance > max_in_row )
            max_in_row = distance;
        }
        if( max_in_row == 0.0 )
          return false;
        weight[ row ] = 1.0 / max_in_row;
      }
      int col;
      for( col = 0; col != element_count; col++ ) {
        for( row = 0; row != col; row++ ) {
          ElementType sum = get( _lu, row, col );
          int index;
          for( index = 0; index != row; index++ )
            sum -= get( _lu, row, index ) * get( _lu, index, col );
          get( _lu, row, col ) = sum;
        }
        int max_rated_row = 0.0;
        ElementType max_rate = 0.0;
        for( row = col; row != element_count; row++ ) {
          ElementType sum = get( _lu, row, col );
          int index;
          for( index = 0; index != col; index++ )
            sum -= get( _lu, row, index ) * get( _lu, index, col );
          get( _lu, row, col ) = sum;
          if( sum < 0.0 )
            sum = -sum;
          ElementType rate = weight[ row ] * sum;
          if( rate >= max_rate ){
            max_rate = rate;
            max_rated_row = row;
          }
        }
        row = col;
        if( row != max_rated_row ) {
          int index;
          for( index = 0; index != element_count; index++ ) {
            {
              ElementType temp = get( _lu, max_rated_row, index );
              get( _lu, max_rated_row, index ) = get( _lu, row, index );
              get( _lu, row, index ) = temp;
            }
          }
          {
            unsigned int temp = _index[ row ];
            _index[ row ] = _index[ max_rated_row ];
            _index[ max_rated_row ] = temp;
          }
          weight[ max_rated_row ] = weight[ row ];
        }
        if( get( _lu, row, row ) == 0.0 )
          return false;
        if( row != element_count - 1 ) {
          ElementType temp = 1.0 / get( _lu, row, row );
          for( row = col + 1; row < element_count; row++ )
            get( _lu, row, col ) *= temp;
        }
      }
      return true;
    }
}

#endif
