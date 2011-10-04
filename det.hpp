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

#ifndef EMATH_DET_HEADER
#define EMATH_DET_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#define EPP_DET2_FUNCTION( index, type ) \
  EPP_DEVICE double det ( \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  ) { \
    return \
      EPP_CAST( double, _source.EPP_ELEMENT( 0 ).EPP_ELEMENT( 0 ) ) * _source.EPP_ELEMENT( 1 ).EPP_ELEMENT( 1 ) - \
      EPP_CAST( double, _source.EPP_ELEMENT( 0 ).EPP_ELEMENT( 1 ) ) * _source.EPP_ELEMENT( 1 ).EPP_ELEMENT( 0 ); \
  }

#define EPP_DET2_PROTOTYPE( index, type ) \
  EPP_DEVICE double det ( \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  );

#define EPP_DET2( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DET2_FUNCTION( 2, type ), \
    \
  )

#define EPP_DET2_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DET2_PROTOTYPE( 2, type ), \
    \
  )


#define EPP_DET_ARGS_WIDTH( args ) \
  BOOST_PP_ARRAY_ELEM( 0, args )

#define EPP_DET_ARGS_SKIP( args ) \
  BOOST_PP_ARRAY_ELEM( 1, args )

#define EPP_SHIFTED_MATRIX_ELEMENT1( index, width ) \
  EPP_ELEMENT( BOOST_PP_DIV( index, width ) ).EPP_ELEMENT( BOOST_PP_ADD( BOOST_PP_MOD( index, width ), 1 ) )

#define EPP_SHIFTED_MATRIX_ELEMENT2( index, width ) \
  EPP_ELEMENT( BOOST_PP_ADD( BOOST_PP_DIV( index, width ), 1 ) ).EPP_ELEMENT( BOOST_PP_ADD( BOOST_PP_MOD( index, width ), 1 ) )

#define EPP_DET3_OPERATION( z, index, args ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( BOOST_PP_DIV( index, EPP_DET_ARGS_WIDTH( args ) ), EPP_DET_ARGS_SKIP( args ) ), \
    temp.EPP_MATRIX_ELEMENT( index, EPP_DET_ARGS_WIDTH( args ) ) = _source.EPP_SHIFTED_MATRIX_ELEMENT1( index, EPP_DET_ARGS_WIDTH( args ) );, \
    temp.EPP_MATRIX_ELEMENT( index, EPP_DET_ARGS_WIDTH( args ) ) = _source.EPP_SHIFTED_MATRIX_ELEMENT2( index, EPP_DET_ARGS_WIDTH( args ) ); \
  )

#define EPP_DET3_FUNCTION( index, type ) \
  EPP_DEVICE double det ( \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  ) { \
    EPP_MATRIX_TYPE( type, BOOST_PP_SUB( index, 1 ) ) temp; \
    double sum = 0.0; \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 0 ) ) \
    ) \
    sum += det( temp ) * _source.EPP_MATRIX_ELEMENT( 0, index ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 1 ) ) \
    ) \
    sum -= det( temp ) * _source.EPP_MATRIX_ELEMENT( index, index ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 2 ) ) \
    ) \
    sum += det( temp ) * _source.EPP_MATRIX_ELEMENT( BOOST_PP_MUL( index, 2 ), index );; \
    return sum; \
  }

#define EPP_DET3_PROTOTYPE( index, type ) \
  EPP_DEVICE double det ( \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  );

#define EPP_DET3( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DET3_FUNCTION( 3, type ), \
    \
  )

#define EPP_DET3_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DET3_PROTOTYPE( 3, type ), \
    \
  )


#define EPP_DET4_FUNCTION( index, type ) \
  EPP_DEVICE double det ( \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  ) { \
    EPP_MATRIX_TYPE( type, BOOST_PP_SUB( index, 1 ) ) temp; \
    double sum = 0.0; \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 0 ) ) \
    ) \
    sum += det( temp ) * _source.EPP_MATRIX_ELEMENT( 0, index ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 1 ) ) \
    ) \
    sum -= det( temp ) * _source.EPP_MATRIX_ELEMENT( index, index ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 2 ) ) \
    ) \
    sum += det( temp ) * _source.EPP_MATRIX_ELEMENT( BOOST_PP_MUL( index, 2 ), index ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_SUB( index, 1 ), BOOST_PP_SUB( index, 1 ) ), \
      EPP_DET3_OPERATION, \
      ( 2, ( BOOST_PP_SUB( index, 1 ), 3 ) ) \
    ) \
    sum -= det( temp ) * _source.EPP_MATRIX_ELEMENT( BOOST_PP_MUL( index, 3 ), index ); \
    return sum; \
  }

#define EPP_DET4_PROTOTYPE( index, type ) \
  EPP_DEVICE double det ( \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  );

#define EPP_DET4( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 3, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DET4_FUNCTION( 4, type ), \
    \
  )

#define EPP_DET4_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 3, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DET4_PROTOTYPE( 4, type ), \
    \
  )

namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DET2, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DET3, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DET4, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DET2_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DET3_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DET4_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif
