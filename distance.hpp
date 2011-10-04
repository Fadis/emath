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

#ifndef EMATH_DISTANCE_HEADER
#define EMATH_DISTANCE_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/addsub.hpp>
#include <emath/dot.hpp>
#include <emath/normalize.hpp>
#include <emath/cross.hpp>
#include <emath/rot90.hpp>
#include <emath/voronoi.hpp>

#define EPP_DISTANCE_POINT_TO_PLANE_FUNCTION( z, index, type ) \
  EPP_DEVICE double pointToPlane ( \
    const EPP_VECTOR_TYPE( type, index ) &_point, \
    const EPP_VECTOR_TYPE( type, index ) &_plane_normal, \
    const EPP_VECTOR_TYPE( type, index ) &_plane_position \
  ) { \
    EPP_VECTOR_TYPE( type, index ) related_position = _point - _plane_position; \
    EPP_VECTOR_TYPE( type, index ) normalized_normal = normalize( _plane_normal ); \
    return dot( normalized_normal, related_position ); \
  }

#define EPP_DISTANCE_POINT_TO_PLANE_PROTOTYPE( z, index, type ) \
  EPP_DEVICE double pointToPlane ( \
    const EPP_VECTOR_TYPE( type, index ) &_point, \
    const EPP_VECTOR_TYPE( type, index ) &_plane_normal, \
    const EPP_VECTOR_TYPE( type, index ) &_plane_position \
  );

#define EPP_DISTANCE_POINT_TO_PLANE( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DISTANCE_POINT_TO_PLANE_FUNCTION, \
    type \
  )

#define EPP_DISTANCE_POINT_TO_PLANE_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DISTANCE_POINT_TO_PLANE_PROTOTYPE, \
    type \
  )



#define EPP_DISTANCE_POINT_TO_LINE_FUNCTION( z, index, type ) \
  EPP_DEVICE double pointToLine ( \
    const EPP_VECTOR_TYPE( type, index ) &_point, \
    const EPP_VECTOR_TYPE( type, index ) &_line_position1, \
    const EPP_VECTOR_TYPE( type, index ) &_line_position2 \
  ) { \
    EPP_VECTOR_TYPE( type, index ) line_direction = _line_position2 - _line_position1; \
    EPP_VECTOR_TYPE( type, index ) normalized_line_direction = normalize( line_direction ); \
    EPP_VECTOR_TYPE( type, index ) related_position = _point - _line_position1; \
    double e = dot( related_position, line_direction ); \
    double f = length( related_position ); \
    return sqrt( f * f - e * e ); \
  }

#define EPP_DISTANCE_POINT_TO_LINE_PROTOTYPE( z, index, type ) \
  EPP_DEVICE double pointToLine ( \
    const EPP_VECTOR_TYPE( type, index ) &_point, \
    const EPP_VECTOR_TYPE( type, index ) &_line_position1, \
    const EPP_VECTOR_TYPE( type, index ) &_line_position2 \
  );

#define EPP_DISTANCE_POINT_TO_LINE( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DISTANCE_POINT_TO_LINE_FUNCTION, \
    type \
  )

#define EPP_DISTANCE_POINT_TO_LINE_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DISTANCE_POINT_TO_LINE_PROTOTYPE, \
    type \
  )


#define EPP_DISTANCE_POINT_TO_POINT_FUNCTION( z, index, type ) \
  EPP_DEVICE double pointToPoint ( \
    const EPP_VECTOR_TYPE( type, index ) &_point1, \
    const EPP_VECTOR_TYPE( type, index ) &_point2 \
  ) { \
    EPP_VECTOR_TYPE( type, index ) related_position = _point2 - _point1; \
    return length( related_position ); \
  }

#define EPP_DISTANCE_POINT_TO_POINT_PROTOTYPE( z, index, type ) \
  EPP_DEVICE double pointToPoint ( \
    const EPP_VECTOR_TYPE( type, index ) &_point1, \
    const EPP_VECTOR_TYPE( type, index ) &_point2 \
  );

#define EPP_DISTANCE_POINT_TO_POINT( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DISTANCE_POINT_TO_POINT_FUNCTION, \
    type \
  )

#define EPP_DISTANCE_POINT_TO_POINT_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DISTANCE_POINT_TO_POINT_PROTOTYPE, \
    type \
  )


#define EPP_DISCERN_AREA_FUNCTION( index, type ) \
  EPP_DEVICE int discernArea( \
    const EPP_VECTOR_TYPE( type, index ) &_current_position, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) *_voronoi ) { \
      EPP_VECTOR_TYPE( type, index ) key_array[ 3 ]; \
      key_array[ 0 ] = _key0; \
      key_array[ 1 ] = _key1; \
      key_array[ 2 ] = _key2; \
      int edge_index; \
      int vertex_index; \
      EPP_VECTOR_TYPE( type, index ) relative_position; \
        for( edge_index = 0; edge_index != 3; edge_index++ ) { \
          relative_position = _current_position - key_array[ edge_index ]; \
          double distance = dot( _voronoi[ edge_index ], relative_position ); \
          if( distance > 0. ) { \
            for( vertex_index = 0; vertex_index != 2; vertex_index++ ) { \
              relative_position = _current_position - key_array[ ( edge_index + vertex_index ) % 3 ]; \
              distance = dot( _voronoi[ edge_index + 3 ], relative_position ); \
              if( distance > 0. ) \
                return edge_index * 2 + vertex_index; \
            } \
            return ( edge_index * 2 + 2 ) % 6; \
          } \
        } \
        return -1; \
    }

#define EPP_DISCERN_AREA_PROTOTYPE( index, type ) \
  EPP_DEVICE int discernArea( \
    const EPP_VECTOR_TYPE( type, index ) &_current_position, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) *_voronoi );

#define EPP_DISCERN_AREA( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISCERN_AREA_FUNCTION( 2, type ), \
    \
  ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISCERN_AREA_FUNCTION( 3, type ), \
    \
  )

#define EPP_DISCERN_AREA_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISCERN_AREA_PROTOTYPE( 2, type ), \
    \
  ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISCERN_AREA_PROTOTYPE( 3, type ), \
    \
  )


#define EPP_DISTANCE_POINT_TO_TRIANGLE_FUNCTION( index, type ) \
  EPP_DEVICE double pointToTriangle( \
    const EPP_VECTOR_TYPE( type, index ) &_current_position, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) &_plane_normal, \
    const EPP_VECTOR_TYPE( type, index ) *_voronoi ) { \
    int position = discernArea( _current_position, _key0, _key1, _key2, _voronoi ); \
    double distance = 0.; \
    if( position == 0 ) \
      distance = pointToPoint( _current_position, _key0 ); \
    else if( position == 1 ) \
      distance = pointToLine( _current_position, _key0, _key1 ); \
    else if( position == 2 ) \
      distance = pointToPoint( _current_position, _key1 ); \
    else if( position == 3 ) \
      distance = pointToLine( _current_position, _key1, _key2 ); \
    else if( position == 4 ) \
      distance = pointToPoint( _current_position, _key2 ); \
    else if( position == 5 ) \
      distance = pointToLine( _current_position, _key2, _key0 ); \
    else if( position == -1 ) \
      distance = pointToPlane( _current_position, _plane_normal, _key0 ); \
    return distance; \
  }

#define EPP_DISTANCE_POINT_TO_TRIANGLE_PROTOTYPE( index, type ) \
  EPP_DEVICE double pointToTriangle( \
    const EPP_VECTOR_TYPE( type, index ) &_current_position, \
    const EPP_VECTOR_TYPE( type, index ) &_key0, \
    const EPP_VECTOR_TYPE( type, index ) &_key1, \
    const EPP_VECTOR_TYPE( type, index ) &_key2, \
    const EPP_VECTOR_TYPE( type, index ) &_plane_normal, \
    const EPP_VECTOR_TYPE( type, index ) *_voronoi );

#define EPP_DISTANCE_POINT_TO_TRIANGLE( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISTANCE_POINT_TO_TRIANGLE_FUNCTION( 2, type ), \
    \
  ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISTANCE_POINT_TO_TRIANGLE_FUNCTION( 3, type ), \
    \
  )

#define EPP_DISTANCE_POINT_TO_TRIANGLE_PROTOTYPES( z, type, unused ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 1, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISTANCE_POINT_TO_TRIANGLE_PROTOTYPE( 2, type ), \
    \
  ) \
  BOOST_PP_IF( \
    BOOST_PP_LESS( 2, EPP_VECTOR_TYPE_MAX_SIZE( type ) ), \
    EPP_DISTANCE_POINT_TO_TRIANGLE_PROTOTYPE( 3, type ), \
    \
  )

namespace emath {

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_PLANE, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_LINE, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_POINT, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISCERN_AREA, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_TRIANGLE, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_PLANE_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_LINE_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_POINT_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISCERN_AREA_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DISTANCE_POINT_TO_TRIANGLE_PROTOTYPES, EPP_UNUSED )
#endif

}

#endif
