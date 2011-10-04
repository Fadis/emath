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

#ifndef EMATH_SCALE_HEADER
#define EMATH_SCALE_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#define EPP_SCALE_ARGS_OPERATOR( args ) \
  BOOST_PP_ARRAY_ELEM( 0, args )

#define EPP_SCALE_ARGS_WIDTH( args ) \
  BOOST_PP_ARRAY_ELEM( 1, args )

#define EPP_SCALE_ARGS_TYPE( args ) \
  BOOST_PP_ARRAY_ELEM( 1, args )

#define EPP_SCALE_VECTOR_OPERATION_2OP( z, index, oper ) \
  dest.EPP_ELEMENT( index ) = \
    _left.EPP_ELEMENT( index ) oper \
    _right;

#define EPP_SCALE_VECTOR_OPERATION_1OP( z, index, oper ) \
  _left.EPP_ELEMENT( index ) oper \
    _right;

#define EPP_SCALE_MATRIX_OPERATION_2OP( z, index, args ) \
  dest.EPP_MATRIX_ELEMENT( index, EPP_SCALE_ARGS_WIDTH( args ) ) = \
    _left.EPP_MATRIX_ELEMENT( index, EPP_SCALE_ARGS_WIDTH( args ) ) EPP_SCALE_ARGS_OPERATOR( args ) \
    _right;

#define EPP_SCALE_MATRIX_OPERATION_1OP( z, index, args ) \
  _left.EPP_MATRIX_ELEMENT( index, EPP_SCALE_ARGS_WIDTH( args ) ) EPP_SCALE_ARGS_OPERATOR( args ) \
    _right;





#define EPP_SCALE_VECTOR_FUNCTION_2OP( z, index, args ) \
  EPP_DEVICE EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    const EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    const EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  ) { \
    EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) dest; \
    BOOST_PP_REPEAT( \
      index, \
      EPP_SCALE_VECTOR_OPERATION_2OP, \
      EPP_SCALE_ARGS_OPERATOR( args ) \
    ) \
    return dest; \
  }

#define EPP_SCALE_VECTOR_PROTOTYPE_2OP( z, index, args ) \
  EPP_DEVICE EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    const EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    const EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  );

#define EPP_SCALE_VECTOR_2OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_VECTOR_FUNCTION_2OP, \
    ( 2, ( oper, type ) ) \
  )

#define EPP_SCALE_VECTOR_PROTOTYPES_2OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_VECTOR_PROTOTYPE_2OP, \
    ( 2, ( oper, type ) ) \
  )


#define EPP_SCALE_VECTOR_FUNCTION_1OP( z, index, args ) \
  EPP_DEVICE EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  ) { \
    BOOST_PP_REPEAT( \
      index, \
      EPP_SCALE_VECTOR_OPERATION_1OP, \
      EPP_SCALE_ARGS_OPERATOR( args ) \
    ) \
    return _left; \
  }

#define EPP_SCALE_VECTOR_PROTOTYPE_1OP( z, index, args ) \
  EPP_DEVICE EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    EPP_VECTOR_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  );

#define EPP_SCALE_VECTOR_1OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_VECTOR_FUNCTION_1OP, \
    ( 2, ( oper, type ) ) \
  )

#define EPP_SCALE_VECTOR_PROTOTYPES_1OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_VECTOR_PROTOTYPE_1OP, \
    ( 2, ( oper, type ) ) \
  )


#define EPP_SCALE_MATRIX_FUNCTION_2OP( z, index, args ) \
  EPP_DEVICE EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    const EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  ) { \
    EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) dest; \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( index, index ), \
      EPP_SCALE_MATRIX_OPERATION_2OP, \
      ( 2, ( EPP_SCALE_ARGS_OPERATOR( args ), index ) ) \
    ) \
    return dest; \
  }

#define EPP_SCALE_MATRIX_PROTOTYPE_2OP( z, index, args ) \
  EPP_DEVICE EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    const EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  );

#define EPP_SCALE_MATRIX_2OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_MATRIX_FUNCTION_2OP, \
    ( 2, ( oper, type ) ) \
  )

#define EPP_SCALE_MATRIX_PROTOTYPES_2OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_MATRIX_PROTOTYPE_2OP, \
    ( 2, ( oper, type ) ) \
  )


#define EPP_SCALE_MATRIX_FUNCTION_1OP( z, index, args ) \
  EPP_DEVICE EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  ) { \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( index, index ), \
      EPP_SCALE_MATRIX_OPERATION_1OP, \
      ( 2, ( EPP_SCALE_ARGS_OPERATOR( args ), index ) ) \
    ) \
    return _left; \
  }

#define EPP_SCALE_MATRIX_PROTOTYPE_1OP( z, index, args ) \
  EPP_DEVICE EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &operator EPP_SCALE_ARGS_OPERATOR( args ) ( \
    EPP_MATRIX_TYPE( EPP_SCALE_ARGS_TYPE( args ), index ) &_left, \
    EPP_SCALAR_TYPE( EPP_SCALE_ARGS_TYPE( args ) ) _right \
  );

#define EPP_SCALE_MATRIX_1OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_MATRIX_FUNCTION_1OP, \
    ( 2, ( oper, type ) ) \
  )

#define EPP_SCALE_MATRIX_PROTOTYPES_1OP( z, type, oper ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_SCALE_MATRIX_PROTOTYPE_1OP, \
    ( 2, ( oper, type ) ) \
  )


namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_2OP, * )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_2OP, / )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_1OP, *= )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_1OP, /= )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_2OP, * )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_2OP, / )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_1OP, *= )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_1OP, /= )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_PROTOTYPES_2OP, * )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_PROTOTYPES_2OP, / )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_PROTOTYPES_1OP, *= )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_SCALE_VECTOR_PROTOTYPES_1OP, /= )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_PROTOTYPES_2OP, * )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_PROTOTYPES_2OP, / )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_PROTOTYPES_1OP, *= )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_SCALE_MATRIX_PROTOTYPES_1OP, /= )
#endif
}

#endif
