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

#ifndef EMATH_VORONOI_HEADER
#define EMATH_VORONOI_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/addsub.hpp>
#include <emath/normalize.hpp>
#include <emath/cross.hpp>
#include <emath/rot90.hpp>

#define EPP_VORONOI_3D_OPERATION_PHASE1( z, index, unused ) \
    _voronoi[ index ] = BOOST_PP_CAT( _key, BOOST_PP_MOD( BOOST_PP_ADD( index, 1 ), 3 ) ) - BOOST_PP_CAT( _key, index ); \
    _voronoi[ index ] = cross( _voronoi[ index ], plane_normal ); \
    _voronoi[ index ] = normalize( _voronoi[ index ] );

#define EPP_VORONOI_3D_OPERATION_PHASE2( z, index, unused ) \
    _voronoi[ BOOST_PP_ADD( index, 3 ) ] = cross( _voronoi[ index ], plane_normal ); \
    _voronoi[ BOOST_PP_ADD( index, 3 ) ] = normalize( _voronoi[ BOOST_PP_ADD( index, 3 ) ] );


#define EPP_VORONOI_3D_FUNCTION( index, type ) \
  EPP_DEVICE void voronoi ( \
    EPP_VECTOR_TYPE( type, index ) *_voronoi, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) &plane_normal \
  ) { \
    BOOST_PP_REPEAT( 3, EPP_VORONOI_3D_OPERATION_PHASE1, EPP_UNUSED ) \
    BOOST_PP_REPEAT( 3, EPP_VORONOI_3D_OPERATION_PHASE2, EPP_UNUSED ) \
  }

#define EPP_VORONOI_3D_PROTOTYPE( index, type ) \
  EPP_DEVICE void voronoi ( \
  EPP_VECTOR_TYPE( type, index ) *_voronoi, \
  const EPP_VECTOR_TYPE( type, index ) &_key0, \
  const EPP_VECTOR_TYPE( type, index ) &_key1, \
  const EPP_VECTOR_TYPE( type, index ) &_key2, \
  const EPP_VECTOR_TYPE( type, index ) &plane_normal \
  );

#define EPP_VORONOI_3D( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_VORONOI_3D_FUNCTION( 3, type ), \
    \
  )

#define EPP_VORONOI_3D_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_VORONOI_3D_PROTOTYPE( 3, type ), \
    \
  )


#define EPP_VORONOI_2D_OPERATION_PHASE1( z, index, unused ) \
  _voronoi[ BOOST_PP_ADD( index, 3 ) ] = BOOST_PP_CAT( _key, BOOST_PP_MOD( BOOST_PP_ADD( index, 1 ), 3 ) ) - BOOST_PP_CAT( _key, index );

#define EPP_VORONOI_2D_OPERATION_PHASE2( z, index, unused ) \
  _voronoi[ index ] = rot90( _voronoi[ BOOST_PP_ADD( index, 3 ) ], -1 );

#define EPP_VORONOI_2D_FUNCTION( index, type ) \
  EPP_DEVICE void voronoi ( \
    EPP_VECTOR_TYPE( type, index ) *_voronoi, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) &plane_normal \
  ) { \
    BOOST_PP_REPEAT( 3, EPP_VORONOI_2D_OPERATION_PHASE1, EPP_UNUSED ) \
    BOOST_PP_REPEAT( 3, EPP_VORONOI_2D_OPERATION_PHASE2, EPP_UNUSED ) \
  }

#define EPP_VORONOI_2D_PROTOTYPE( index, type ) \
  EPP_DEVICE void voronoi ( \
    EPP_VECTOR_TYPE( type, index ) *_voronoi, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) &plane_normal \
  );

#define EPP_VORONOI_2D( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_VORONOI_2D_FUNCTION( 2, type ), \
    \
  )

#define EPP_VORONOI_2D_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_VORONOI_2D_PROTOTYPE( 2, type ), \
    \
  )

namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_VORONOI_3D, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_VORONOI_2D, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_VORONOI_3D_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_VORONOI_2D_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif
