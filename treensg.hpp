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

#ifndef ENSG_TREENSG_HEADER
#define ENSG_TREENSG_HEADER


#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/scale.hpp>
#include <emath/multiarray.hpp>
#include <emath/mps.hpp>

namespace emath {

#ifdef __CUDACC__
  namespace treensg {
    EPP_DEVICE unsigned int getLinearIndex() {
      return blockIdx.x * blockDim.x + threadIdx.x;
    }

    EPP_DEVICE void clearTNSG(
      unsigned int capacity,
      unsigned int *buffer,
      unsigned int _index
    ) {
      int counter;
      for( counter = 0; counter != capacity; counter++ )
        buffer[ capacity * _index + counter ] = 0;
    }
    
    extern "C"
    EPP_GLOBAL void clearTNSGAll(
      unsigned int capacity,
      unsigned int *buffer
    ) {
      unsigned int self = getLinearIndex();
      clearTNSG( capacity, buffer, self );
    }
    
    EPP_DEVICE void clearTNSGOutput(
      unsigned int capacity,
      unsigned int *output,
      unsigned int _index
    ) {
      int counter;
      for( counter = 0; counter != 27 * capacity; counter++ )
        output[ capacity * 27 * _index + counter ] = 0;
    }
    
    extern "C"
    EPP_GLOBAL void clearTNSGOutputAll(
      unsigned int capacity,
      unsigned int *output
    ) {
      unsigned int self = getLinearIndex();
      clearTNSGOutput( capacity, output, self );
    }
    
    EPP_DEVICE uint3 hash( uint3 size, int3 value ) {
      uint3 temp;
      int3 scaled_value = value;
      temp.x = scaled_value.x % size.x;
      temp.y = scaled_value.y % size.y;
      temp.z = scaled_value.z % size.z;
      return temp;
    }
    
    EPP_DEVICE void clearPrimaryMap(
      unsigned int *primary_map,
      unsigned int _index
    ) {
      primary_map[ _index ] = 0;
    }

    extern "C"
    EPP_GLOBAL void clearPrimaryMapAll(
      unsigned int *primary_map
    ) {
      unsigned int self = getLinearIndex();
      clearPrimaryMap( primary_map, self );
    }

    EPP_DEVICE void clearSecondaryMap(
      unsigned int *secondary_map,
      unsigned int _index
    ) {
      secondary_map[ _index ] = 0;
    }

    extern "C"
    EPP_GLOBAL void clearSecondaryMapAll(
      unsigned int *secondary_map
    ) {
      unsigned int self = getLinearIndex();
      clearSecondaryMap( secondary_map, self );
    }

    EPP_DEVICE void detectPrimaryActiveBlock(
      unsigned int *primary_map,
      float boxel_size,
      const float3 &_position,
      TreeLocationCache *_location_cache
    ) {
      uint3 primary_position_in_grid = emath::vector_cast< uint3 >( _position / boxel_size );
      _location_cache->pos = primary_position_in_grid;
      primary_position_in_grid.x >>= 6;
      primary_position_in_grid.y >>= 4;
      primary_position_in_grid.z >>= 2;
      primary_position_in_grid.x &= 0x03;
      primary_position_in_grid.y &= 0x06;
      primary_position_in_grid.z &= 0x0C;

      unsigned int linear_pos =
        primary_position_in_grid.x |
        primary_position_in_grid.y |
        primary_position_in_grid.z;
      primary_map[ linear_pos ] = 1;
//      atomicOr( primary_map + linear_pos, 1 );
      _location_cache->primary_linear_pos = linear_pos;
    }

    extern "C"
    EPP_GLOBAL void detectPrimaryActiveBlockAll(
      unsigned int *primary_map,
      float boxel_size,
      Particle *_particles,
      TreeLocationCache *_location_cache
    ) {
      unsigned int self = getLinearIndex();
      detectPrimaryActiveBlock( primary_map, boxel_size, _particles[ self ].position, _location_cache + self );
    }

    EPP_DEVICE void calcPrimaryOffset(
      unsigned int *primary_map,
      unsigned int *sum,
      unsigned int _index
    ) {
      unsigned int value = primary_map[ _index ];
      if( value )
        primary_map[ _index ] = atomicAdd( sum, value );
    }

    extern "C"
    EPP_GLOBAL void calcPrimaryOffsetAll(
      unsigned int *primary_map,
      unsigned int *sum
    ) {
      unsigned int self = getLinearIndex();
      calcPrimaryOffset( primary_map, sum, self );
    }


    EPP_DEVICE void detectSecondaryActiveBlock(
      unsigned int *primary_map,
      unsigned int *secondary_map,
      float boxel_size,
      TreeLocationCache *_location_cache
    ) {
      uint3 secondary_position_in_grid = _location_cache->pos;
      secondary_position_in_grid.x >>= 3;
      secondary_position_in_grid.z <<= 3;
      secondary_position_in_grid.x &= 0x07;
      secondary_position_in_grid.y &= 0x38;
      secondary_position_in_grid.z &= 0x1C0;
      unsigned int primary_linear_pos = _location_cache->primary_linear_pos;
      unsigned int secondary_linear_pos =
        ( primary_map[ primary_linear_pos ] << 9 ) |
        secondary_position_in_grid.x |
        secondary_position_in_grid.y |
        secondary_position_in_grid.z;
      secondary_map[ secondary_linear_pos ] = 1;
//      atomicOr( secondary_map + secondary_linear_pos, 1 );
      _location_cache->secondary_linear_pos = secondary_linear_pos;
    }

    extern "C"
    EPP_GLOBAL void detectSecondaryActiveBlockAll(
      unsigned int *primary_map,
      unsigned int *secondary_map,
      float boxel_size,
      TreeLocationCache *_location_cache
    ) {
      unsigned int self = getLinearIndex();
      detectSecondaryActiveBlock( primary_map, secondary_map, boxel_size, _location_cache + self );
    }

    EPP_DEVICE void calcSecondaryOffset(
      unsigned int *secondary_map,    
      unsigned int *sum,
      unsigned int _index
    ) {
      unsigned int value = secondary_map[ _index ];
      if( value )
        secondary_map[ _index ] = atomicAdd( sum, value );
    }

    extern "C"
    EPP_GLOBAL void calcSecondaryOffsetAll(
      unsigned int *secondary_map,
      unsigned int *sum
    ) {
      unsigned int self = getLinearIndex();
      calcSecondaryOffset( secondary_map, sum, self );
    }

    EPP_DEVICE void setTNSG(
      float boxel_size, unsigned int capacity,
      unsigned int *buffer,
      const float3 &_position, unsigned int _index,
      unsigned int *primary_map,
      unsigned int *secondary_map,
      TreeLocationCache *_location_cache
    ) {

      uint3 final_position_in_grid = _location_cache->pos;
      final_position_in_grid.y <<= 3;
      final_position_in_grid.z <<= 6;
      final_position_in_grid.x &= 0x07;
      final_position_in_grid.y &= 0x38;
      final_position_in_grid.z &= 0x1C0;

      unsigned int secondary_linear_pos = _location_cache->secondary_linear_pos;
      unsigned int where_to_set = (
        ( secondary_map[ secondary_linear_pos ] << 9 ) |
        final_position_in_grid.x |
        final_position_in_grid.y |
        final_position_in_grid.z ) * capacity;
      int element_index;
      _index++;
      for( element_index = 0; element_index != capacity; element_index++ )
        if( !atomicCAS( buffer + where_to_set + element_index, 0, _index ) )
          break;
    }

    extern "C"
    EPP_GLOBAL void setTNSGAll(
      float boxel_size, unsigned int capacity,
      unsigned int *buffer,
      Particle *_particles,
      unsigned int *primary_map,
      unsigned int *secondary_map,
      TreeLocationCache *_location_cache
    ) {
        unsigned int self = getLinearIndex();
        setTNSG( boxel_size, capacity, buffer, _particles[ self ].position, self, primary_map, secondary_map, _location_cache + self );
    }




/*
    EPP_DEVICE void getTNSG4debug(
      float boxel_size, unsigned int capacity,
      unsigned int *buffer,
      unsigned int *_output,
      unsigned int _self, const float3 &_position,
      unsigned int *primary_map,
      unsigned int *secondary_map,
      unsigned int *debug_storage
    ) {
      int3 position_in_grid_original = emath::vector_cast< int3 >( _position / boxel_size );
      unsigned int offset_index;
      for( offset_index = 0; offset_index != 27; offset_index++ ) {
        int3 primary_position_in_grid = position_in_grid_original + offsets[ offset_index ];
        primary_position_in_grid.x /= 64;
        primary_position_in_grid.y /= 64;
        primary_position_in_grid.z /= 64;
        primary_position_in_grid.x = static_cast< unsigned int >( primary_position_in_grid.x ) % 8;
        primary_position_in_grid.y = static_cast< unsigned int >( primary_position_in_grid.y ) % 8;
        primary_position_in_grid.z = static_cast< unsigned int >( primary_position_in_grid.z ) % 8;

        int3 secondary_position_in_grid = position_in_grid_original + offsets[ offset_index ];
        secondary_position_in_grid.x /= 8;
        secondary_position_in_grid.y /= 8;
        secondary_position_in_grid.z /= 8;
        secondary_position_in_grid.x = static_cast< unsigned int >( secondary_position_in_grid.x ) % 8;
        secondary_position_in_grid.y = static_cast< unsigned int >( secondary_position_in_grid.y ) % 8;
        secondary_position_in_grid.z = static_cast< unsigned int >( secondary_position_in_grid.z ) % 8;

        int3 final_position_in_grid = position_in_grid_original + offsets[ offset_index ];
        final_position_in_grid.x = static_cast< unsigned int >( final_position_in_grid.x ) % 8;
        final_position_in_grid.y = static_cast< unsigned int >( final_position_in_grid.y ) % 8;
        final_position_in_grid.z = static_cast< unsigned int >( final_position_in_grid.z ) % 8;

        unsigned int primary_linear_pos =
          primary_position_in_grid.x +
          primary_position_in_grid.y * 8 +
          primary_position_in_grid.z * 64;
        unsigned int secondary_linear_pos =
          primary_map[ primary_linear_pos ] * 512 +
          secondary_position_in_grid.x +
          secondary_position_in_grid.y * 8 +
          secondary_position_in_grid.z * 64;
        unsigned int where_to_read = (
          secondary_map[ secondary_linear_pos ] * 512 +
          final_position_in_grid.x +
          final_position_in_grid.y * 8 +
          final_position_in_grid.z * 64 ) * capacity;
        int element_index;
        debug_storage[ 0 ] = primary_position_in_grid.x;
        debug_storage[ 1 ] = secondary_position_in_grid.x;
        debug_storage[ 2 ] = final_position_in_grid.x;
        debug_storage[ 3 ] = position_in_grid_original.x;
        debug_storage[ 4 ] = position_in_grid_original.y;
        debug_storage[ 5 ] = position_in_grid_original.z;
      }
    }

    extern "C"
    EPP_GLOBAL void getTNSGAll4debug(
      float boxel_size, unsigned int capacity,
      unsigned int *buffer,
      unsigned int *_output,
      Particle *_particles,
      unsigned int *primary_map,
      unsigned int *secondary_map,
      unsigned int *debug_storage
    ) {
      unsigned int self = getLinearIndex();
      getTNSG4debug( boxel_size, capacity, buffer, _output + ( capacity * 27 * self ), self, _particles[ self ].position, primary_map, secondary_map, debug_storage + self * 8 );
    }
*/
    EPP_DEVICE void getTNSG(
      float boxel_size, unsigned int capacity,
      unsigned int *buffer,
      unsigned int *_output,
      unsigned int _self, const float3 &_position,
      unsigned int *primary_map,
      unsigned int *secondary_map
    ) {
      int3 position_in_grid_original = emath::vector_cast< int3 >( _position / boxel_size );
      unsigned int offset_index;
      for( offset_index = 0; offset_index != 27; offset_index++ ) {
        int3 primary_position_in_grid = position_in_grid_original + offsets[ offset_index ];
        primary_position_in_grid.x >>= 6;
        primary_position_in_grid.y >>= 4;
        primary_position_in_grid.z >>= 2;
        primary_position_in_grid.x = static_cast< unsigned int >( primary_position_in_grid.x ) & 0x03;
        primary_position_in_grid.y = static_cast< unsigned int >( primary_position_in_grid.y ) & 0x06;
        primary_position_in_grid.z = static_cast< unsigned int >( primary_position_in_grid.z ) & 0x0C;
        unsigned int linear_pos =
          primary_position_in_grid.x |
          primary_position_in_grid.y |
          primary_position_in_grid.z;
        linear_pos = primary_map[ linear_pos ] << 9;
        if( !linear_pos )
          continue;
        int3 secondary_position_in_grid = position_in_grid_original + offsets[ offset_index ];
        secondary_position_in_grid.x >>= 3;
        secondary_position_in_grid.z <<= 3;
        secondary_position_in_grid.x = static_cast< unsigned int >( secondary_position_in_grid.x ) & 0x07;
        secondary_position_in_grid.y = static_cast< unsigned int >( secondary_position_in_grid.y ) & 0x38;
        secondary_position_in_grid.z = static_cast< unsigned int >( secondary_position_in_grid.z ) & 0x1C0;
        linear_pos |=
          secondary_position_in_grid.x |
          secondary_position_in_grid.y |
          secondary_position_in_grid.z;
        linear_pos = secondary_map[ linear_pos ] << 9;
        if( !linear_pos )
          continue;
        int3 final_position_in_grid = position_in_grid_original + offsets[ offset_index ];
        final_position_in_grid.y <<= 3;
        final_position_in_grid.z <<= 6;
        final_position_in_grid.x = static_cast< unsigned int >( final_position_in_grid.x ) & 0x07;
        final_position_in_grid.y = static_cast< unsigned int >( final_position_in_grid.y ) & 0x38;
        final_position_in_grid.z = static_cast< unsigned int >( final_position_in_grid.z ) & 0x1C0;
        linear_pos = (
          linear_pos |
          final_position_in_grid.x |
          final_position_in_grid.y |
          final_position_in_grid.z ) * capacity;
        int element_index;
        for( element_index = 0; element_index != capacity - 1;  element_index++ ) {
          unsigned int value = buffer[ linear_pos + element_index ];
          if( value ) {
            if( value != _self + 1 ) {
              *_output = value;
              _output++;
            }
          }
          else
            break;
        }
      }
      *_output = 0;
    }

    extern "C"
    EPP_GLOBAL void getTNSGAll(
      float boxel_size, unsigned int capacity,
      unsigned int *buffer,
      unsigned int *_output,
      Particle *_particles,
      unsigned int *primary_map,
      unsigned int *secondary_map
    ) {
      unsigned int self = getLinearIndex();
      getTNSG( boxel_size, capacity, buffer, _output + ( capacity * 27 * self ), self, _particles[ self ].position, primary_map, secondary_map );
    }
  }
#endif
}
#endif
