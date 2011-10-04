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

#ifndef EMATH_ROT90_HEADER
#define EMATH_ROT90_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>


#define EPP_ROT90_2D_FUNCTION( index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) rot90 ( \
    const EPP_VECTOR_TYPE( type, index ) &_source, \
    int _rot \
  ) { \
    EPP_VECTOR_TYPE( type, index ) dest = _source; \
    EPP_SCALAR_TYPE( type ) temp; \
    if( _rot > 0 ) { \
      for( ; _rot; _rot-- ) { \
        temp = dest.EPP_ELEMENT( 1 ); \
        dest.EPP_ELEMENT( 1 ) = dest.EPP_ELEMENT( 0 ); \
        dest.EPP_ELEMENT( 0 ) = -temp; \
      } \
    } \
    else { \
      for( ; _rot; _rot++ ) { \
        temp = dest.EPP_ELEMENT( 0 ); \
        dest.EPP_ELEMENT( 0 ) = dest.EPP_ELEMENT( 1 ); \
        dest.EPP_ELEMENT( 1 ) = -temp; \
      } \
    } \
    return dest; \
  }

#define EPP_ROT90_2D_PROTOTYPE( index, type ) \
  EPP_DEVICE EPP_VECTOR_TYPE( type, index ) rot90 ( \
    const EPP_VECTOR_TYPE( type, index ) &_source, \
    int _rot \
  );


#define EPP_ROT90_2D( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_ROT90_2D_FUNCTION( 2, type ), \
    \
  )

#define EPP_ROT90_2D_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_ROT90_2D_PROTOTYPE( 2, type ), \
    \
  )
namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_ROT90_2D, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_ROT90_2D_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif
