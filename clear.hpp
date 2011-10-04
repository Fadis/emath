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

#ifndef EMATH_CLEAR_HEADER
#define EMATH_CLEAR_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#define EPP_CLEAR_VECTOR_OPERATION( z, index, scalar_type ) \
  _source.EPP_ELEMENT( index ) = EPP_CAST( scalar_type, 0.0 );

#define EPP_CLEAR_VECTOR_FUNCTION( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) &clear ( \
    EPP_VECTOR_TYPE( type, index ) &_source \
  ) { \
    BOOST_PP_REPEAT( \
      index, \
      EPP_CLEAR_VECTOR_OPERATION, \
      EPP_SCALAR_TYPE( type ) \
    ) \
    return _source; \
  }

#define EPP_CLEAR_VECTOR_PROTOTYPE( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) &clear ( \
    EPP_VECTOR_TYPE( type, index ) &_source \
  );

#define EPP_CLEAR_VECTOR( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_CLEAR_VECTOR_FUNCTION, \
    type \
  )

#define EPP_CLEAR_VECTOR_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_CLEAR_VECTOR_PROTOTYPE, \
    type \
  )

#define EPP_CLEAR_ARGS_WIDTH( args ) \
  BOOST_PP_ARRAY_ELEM( 0, args )

#define EPP_CLEAR_ARGS_TYPE( args ) \
  BOOST_PP_ARRAY_ELEM( 1, args )

#define EPP_CLEAR_MATRIX_OPERATION( z, index, args ) \
  _source.EPP_MATRIX_ELEMENT( index, EPP_CLEAR_ARGS_WIDTH( args ) ) = EPP_CAST( EPP_SCALAR_TYPE( EPP_CLEAR_ARGS_TYPE( args ) ), 0.0 );

#define EPP_CLEAR_MATRIX_FUNCTION( z, index, type ) \
  EPP_DEVICE EPP_MATRIX_TYPE( type, index ) &clear ( \
    EPP_MATRIX_TYPE( type, index ) &_source \
  ) { \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( index, index ), \
      EPP_CLEAR_MATRIX_OPERATION, \
      ( 2, ( index, type ) ) \
    ) \
    return _source; \
  }

#define EPP_CLEAR_MATRIX_PROTOTYPE( z, index, type ) \
  EPP_DEVICE EPP_MATRIX_TYPE( type, index ) &clear ( \
    EPP_MATRIX_TYPE( type, index ) &_source \
  );

#define EPP_CLEAR_MATRIX( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_CLEAR_MATRIX_FUNCTION, \
    type \
  )

#define EPP_CLEAR_MATRIX_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_CLEAR_MATRIX_PROTOTYPE, \
    type \
  )

namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_CLEAR_VECTOR, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_CLEAR_MATRIX, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_CLEAR_VECTOR_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_CLEAR_MATRIX_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif