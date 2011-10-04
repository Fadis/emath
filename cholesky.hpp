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

#ifndef EMATH_CHOLESKY_HEADER
#define EMATH_CHOLESKY_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

namespace emath {
  template< typename MatrixType >
    EPP_DEVICE bool cholesky( MatrixType &_dest ) {
      typedef typename Traits< typename Traits< MatrixType >::ElementType >::ElementType ElementType;
      const unsigned int element_count = Traits< MatrixType >::element_count;
      int row;
      for ( row = 0; row != element_count; ++row ) {
        double temp = get( _dest, row, row );
        int col;
        for ( col = 0; col != row; ++col )
          temp -= get( _dest, row, col ) * get( _dest, row, col );
        if( temp <= 0 )
          return false;
        get( _dest, row, row ) = sqrt( temp );
        int index;
        for ( index = row; ++index != element_count;  ) {
          double temp = get( _dest, index, row );
          int col;
          for ( col =0; col != row; ++col )
            temp -= get( _dest, index, col ) * get( _dest, row, col );
          get( _dest, index, row ) = temp / get( _dest, row, row );
        }
      }
      return true;
    }

  template< typename MatrixType >
    EPP_DEVICE bool incompleteCholesky( MatrixType &_dest ) {
      typedef typename Traits< typename Traits< MatrixType >::ElementType >::ElementType ElementType;
      const unsigned int element_count = Traits< MatrixType >::element_count;
      int row;
      for ( row = 0; row != element_count; ++row ) {
        double temp = get( _dest, row, row );
        int col;
        for ( col = 0; col != row; ++col )
          temp -= get( _dest, row, col ) * get( _dest, row, col );
        if( temp <= 0 )
          return false;
        get( _dest, row, row ) = sqrt( temp );
        int index;
        for ( index = row; ++index != element_count;  ) {
          if( get( _dest, row, index ) != 0.0 ) {
            double temp = get( _dest, index, row );
            int col;
            for ( col =0; col != row; ++col )
              temp -= get( _dest, index, col ) * get( _dest, row, col );
            get( _dest, index, row ) = temp / get( _dest, row, row );
          }
          else
            get( _dest, index, row ) = 0.0;
        }
      }
      return true;
    }
}

#endif