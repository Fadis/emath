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

#ifndef EMATH_MOD_HEADER
#define EMATH_MOD_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#define EPP_MOD_OPERATION_2OP( z, index, unused ) \
  dest.EPP_ELEMENT( index ) = \
    _left.EPP_ELEMENT( index ) % \
    _right.EPP_ELEMENT( index );

#define EPP_MOD_FUNCTION_2OP( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) operator % ( \
    const EPP_VECTOR_TYPE( type, index ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  ) { \
    EPP_VECTOR_TYPE( type, index ) dest; \
    BOOST_PP_REPEAT( \
      index, \
      EPP_MOD_OPERATION_2OP, \
      EPP_UNUSED \
    ) \
    return dest; \
}

#define EPP_MOD_PROTOTYPE_2OP( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) mod ( \
  const EPP_VECTOR_TYPE( type, index ) &_left, \
  const EPP_VECTOR_TYPE( type, index ) &_right \
  );

#define EPP_MOD_2OP( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MOD_FUNCTION_2OP, \
    type \
  )

#define EPP_MOD_PROTOTYPES_2OP( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MOD_PROTOTYPE_2OP, \
    type \
  )

#define EPP_MOD_OPERATION_1OP( z, index, unused ) \
  _left.EPP_ELEMENT( index ) %= \
    _right.EPP_ELEMENT( index );

#define EPP_MOD_FUNCTION_1OP( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) &operator %= ( \
    EPP_VECTOR_TYPE( type, index ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  ) { \
    BOOST_PP_REPEAT( \
      index, \
      EPP_MOD_OPERATION_1OP, \
      EPP_UNUSED \
    ) \
    return _left; \
}

#define EPP_MOD_PROTOTYPE_1OP( z, index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) &operator %= ( \
    EPP_VECTOR_TYPE( type, index ) &_left, \
    const EPP_VECTOR_TYPE( type, index ) &_right \
  );

#define EPP_MOD_1OP( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MOD_FUNCTION_1OP, \
    type \
  )

#define EPP_MOD_PROTOTYPES_1OP( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_MOD_PROTOTYPE_1OP, \
    type \
  )

namespace emath {

#ifdef __CUDACC__
  BOOST_PP_REPEAT( 9, EPP_MOD_2OP, EPP_UNUSED )
  BOOST_PP_REPEAT( 9, EPP_MOD_1OP, EPP_UNUSED )
#else
  BOOST_PP_REPEAT( 9, EPP_MOD_PROTOTYPES_2OP, EPP_UNUSED )
  BOOST_PP_REPEAT( 9, EPP_MOD_PROTOTYPES_1OP, EPP_UNUSED )
#endif

}

#endif
