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

#ifndef EMATH_MULT_HEADER
#define EMATH_MULT_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/clear.hpp>


#define EPP_MATRIX_VECTOR_MULT_OPERATION( z, index, width ) \
  dest.EPP_ELEMENT( BOOST_PP_DIV( index, width ) ) += \
    _left.EPP_MATRIX_ELEMENT( index, width ) * \
    _right.EPP_ELEMENT( BOOST_PP_MOD( index, width ) );

#define EPP_MATRIX_VECTOR_MULT_FUNCTION( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) operator * ( \
    const EPP_MATRIX_TYPE( type, index ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  ) { \
    EPP_VECTOR_TYPE( type, index ) dest; \
    clear( dest ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( index, index ), \
      EPP_MATRIX_VECTOR_MULT_OPERATION, \
      index \
    ) \
    return dest; \
  }

#define EPP_MATRIX_VECTOR_MULT_PROTOTYPE( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) operator * ( \
    const EPP_MATRIX_TYPE( type, index ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  );

#define EPP_MATRIX_VECTOR_MULT( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MATRIX_VECTOR_MULT_FUNCTION, \
    type \
  )

#define EPP_MATRIX_VECTOR_MULT_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MATRIX_VECTOR_MULT_PROTOTYPE, \
    type \
  )


#define EPP_COORDINATE_MULT_COPY_TO_TEMP( z, index, unused ) \
  temp.EPP_ELEMENT( index ) = \
    _right.EPP_ELEMENT( index );

#define EPP_COORDINATE_MULT_COPY_FROM_TEMP( z, index, unused ) \
  dest.EPP_ELEMENT( index ) = \
    temp.EPP_ELEMENT( index ); \

#define EPP_COORDINATE_MULT_SCALE( z, index, last ) \
  temp.EPP_ELEMENT( index ) /= \
    temp.EPP_ELEMENT( last );

#define EPP_COORDINATE_MULT_FUNCTION( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) operator * ( \
    const EPP_MATRIX_TYPE( type, BOOST_PP_ADD( index, 1 ) ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  ) { \
    EPP_VECTOR_TYPE( type, BOOST_PP_ADD( index, 1 ) ) temp; \
    EPP_VECTOR_TYPE( type, index ) dest; \
    BOOST_PP_REPEAT( \
      index, \
      EPP_COORDINATE_MULT_COPY_TO_TEMP, \
      EPP_UNUSED \
    ) \
    temp.EPP_ELEMENT( index ) = static_cast< EPP_SCALAR_TYPE( type ) >( 1.0 ); \
    temp = _left * temp; \
    BOOST_PP_REPEAT( \
      index, \
      EPP_COORDINATE_MULT_SCALE, \
      index \
    ) \
    BOOST_PP_REPEAT( \
      index, \
      EPP_COORDINATE_MULT_COPY_FROM_TEMP, \
      EPP_UNUSED \
    ) \
    return dest; \
  }

#define EPP_COORDINATE_MULT_PROTOTYPE( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) operator * ( \
    const EPP_MATRIX_TYPE( type, BOOST_PP_ADD( index, 1 ) ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  );

#define EPP_COORDINATE_MULT( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    BOOST_PP_SUB( EPP_VECTOR_TYPE_MAX_SIZE( type ), 1 ), \
    EPP_COORDINATE_MULT_FUNCTION, \
    type \
  )

#define EPP_COORDINATE_MULT_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    BOOST_PP_SUB( EPP_VECTOR_TYPE_MAX_SIZE( type ), 1 ), \
    EPP_COORDINATE_MULT_PROTOTYPE, \
    type \
  )


#define EPP_3DCOUNT_ROW( index, width ) \
  BOOST_PP_DIV( index, BOOST_PP_MUL( width, width ) )

#define EPP_3DCOUNT_COL( index, width ) \
  BOOST_PP_MOD( BOOST_PP_DIV( index, width ), width )

#define EPP_3DCOUNT_COUNT( index, width ) \
  BOOST_PP_MOD( index, width )

#define EPP_MATRIX_MATRIX_MULT_OPERATION( z, index, width ) \
  dest.EPP_ELEMENT( EPP_3DCOUNT_ROW( index, width ) ).EPP_ELEMENT( EPP_3DCOUNT_COL( index, width ) ) += \
    _left.EPP_ELEMENT( EPP_3DCOUNT_ROW( index, width ) ).EPP_ELEMENT( EPP_3DCOUNT_COUNT( index, width ) ) * \
    _right.EPP_ELEMENT( EPP_3DCOUNT_COUNT( index, width ) ).EPP_ELEMENT( EPP_3DCOUNT_COL( index, width ) );

#define EPP_MATRIX_MATRIX_MULT_FUNCTION( z, index, type ) \
  EPP_DEVICE EPP_MATRIX_TYPE( type, index ) operator * ( \
    const EPP_MATRIX_TYPE( type, index ) &_left, \
    const EPP_MATRIX_TYPE( type, index ) &_right \
  ) { \
    EPP_MATRIX_TYPE( type, index ) dest; \
    clear( dest ); \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( BOOST_PP_MUL( index, index ), index ), \
      EPP_MATRIX_MATRIX_MULT_OPERATION, \
      index \
    ) \
    return dest; \
  }

#define EPP_MATRIX_MATRIX_MULT_PROTOTYPE( z, index, type ) \
  EPP_DEVICE EPP_MATRIX_TYPE( type, index ) operator * ( \
    const EPP_MATRIX_TYPE( type, index ) &_left, \
    const EPP_MATRIX_TYPE( type, index ) &_right \
  );

#define EPP_MATRIX_MATRIX_MULT( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MATRIX_MATRIX_MULT_FUNCTION, \
    type \
  )

#define EPP_MATRIX_MATRIX_MULT_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MATRIX_MATRIX_MULT_PROTOTYPE, \
    type \
  )

namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_MATRIX_VECTOR_MULT, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_COORDINATE_MULT, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_MATRIX_MATRIX_MULT, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_MATRIX_VECTOR_MULT_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_COORDINATE_MULT_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_MATRIX_MATRIX_MULT_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif
