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

#ifndef EMATH_IDENTITY_HEADER
#define EMATH_IDENTITY_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#define EPP_IDENTITY_ARGS_WIDTH( args ) \
  BOOST_PP_ARRAY_ELEM( 0, args )

#define EPP_IDENTITY_ARGS_TYPE( args ) \
  BOOST_PP_ARRAY_ELEM( 1, args )

#define EPP_IDENTITY_OPERATION( z, index, args ) \
  _source.EPP_MATRIX_ELEMENT( index, EPP_IDENTITY_ARGS_WIDTH( args ) ) = \
    EPP_CAST( \
      EPP_SCALAR_TYPE( EPP_IDENTITY_ARGS_TYPE( args ) ), \
      BOOST_PP_IF( \
        BOOST_PP_EQUAL( \
          BOOST_PP_MOD( index, EPP_IDENTITY_ARGS_WIDTH( args ) ), \
          BOOST_PP_DIV( index, EPP_IDENTITY_ARGS_WIDTH( args ) ) \
        ), \
        1.0, \
        0.0 \
      ) \
    );

#define EPP_IDENTITY_FUNCTION( z, index, type ) \
  EPP_DEVICE EPP_MATRIX_TYPE( type, index ) &identity ( \
    EPP_MATRIX_TYPE( type, index ) &_source \
  ) { \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( index, index ), \
      EPP_IDENTITY_OPERATION, \
      ( 2, ( index, type ) ) \
    ) \
    return _source; \
  }

#define EPP_IDENTITY_PROTOTYPE( z, index, type ) \
  EPP_DEVICE EPP_MATRIX_TYPE( type, index ) &identity ( \
    EPP_MATRIX_TYPE( type, index ) &_source \
  );

#define EPP_IDENTITY( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_IDENTITY_FUNCTION, \
    type \
  )

#define EPP_IDENTITY_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_IDENTITY_PROTOTYPE, \
    type \
  )

namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_IDENTITY, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_IDENTITY_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif