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

#ifndef ENSG_SLICENSG_HEADER
#define ENSG_SLICENSG_HEADER


#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/scale.hpp>
#include <emath/multiarray.hpp>
#include <emath/mps.hpp>

namespace emath {

#ifdef __CUDACC__
  namespace slicensg {
    EPP_DEVICE unsigned int getLinearIndex() {
      return blockIdx.x * blockDim.x + threadIdx.x;
    }

    EPP_DEVICE void clearSNSG(
      unsigned int capacity,
      unsigned int *buffer,
      unsigned int _index
    ) {
      int counter;
      for( counter = 0; counter != capacity; counter++ )
        buffer[ capacity * _index + counter ] = 0;
    }
    
    extern "C"
    EPP_GLOBAL void clearSNSGAll(
      unsigned int capacity,
      unsigned int *buffer
    ) {
      unsigned int self = getLinearIndex();
      clearSNSG( capacity, buffer, self );
    }
    
    EPP_DEVICE void clearSNSGOutput(
      unsigned int capacity,
      unsigned int *output,
      unsigned int _index
    ) {
      int counter;
      for( counter = 0; counter != 27 * capacity; counter++ )
        output[ capacity * 27 * _index + counter ] = 0;
    }
    
    extern "C"
    EPP_GLOBAL void clearSNSGOutputAll(
      unsigned int capacity,
      unsigned int *output
    ) {
      unsigned int self = getLinearIndex();
      clearSNSGOutput( capacity, output, self );
    }
    
    EPP_DEVICE uint3 hash( uint3 size, int3 value ) {
      uint3 temp;
      int3 scaled_value = value;
      temp.x = scaled_value.x % size.x;
      temp.y = scaled_value.y % size.y;
      temp.z = scaled_value.z % size.z;
      return temp;
    }
    
    EPP_DEVICE void clearMinMax(
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      SliceIndex *minmax,
      unsigned int _index
    ) {
      minmax[ _index ].x_min = x_boxel_count;
      minmax[ _index ].y_min = y_boxel_count;
      minmax[ _index ].x_max = 0;
      minmax[ _index ].y_max = 0;
    }

    extern "C"
    EPP_GLOBAL void clearMinMaxAll(
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      SliceIndex *minmax
    ) {
      unsigned int self = getLinearIndex();
      clearMinMax( x_boxel_count, y_boxel_count, minmax, self );
    }


    EPP_DEVICE void detectMinMax(
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      unsigned int z_boxel_count,
      SliceIndex *minmax,
      float boxel_size,
      const float3 &_position
    ) {
      int3 position_in_grid = emath::vector_cast< int3 >( _position / boxel_size );
      position_in_grid.x = static_cast< unsigned int >( position_in_grid.x ) % x_boxel_count;
      position_in_grid.y = static_cast< unsigned int >( position_in_grid.y ) % y_boxel_count;
      position_in_grid.z = static_cast< unsigned int >( position_in_grid.z ) % z_boxel_count;
      atomicMin( &( minmax[ position_in_grid.z ].x_min ), position_in_grid.x );
      atomicMax( &( minmax[ position_in_grid.z ].x_max ), position_in_grid.x );
      atomicMin( &( minmax[ position_in_grid.z ].y_min ), position_in_grid.y );
      atomicMax( &( minmax[ position_in_grid.z ].y_max ), position_in_grid.y );
    }

    extern "C"
    EPP_GLOBAL void detectMinMaxAll(
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      unsigned int z_boxel_count,
      SliceIndex *minmax,
      float boxel_size,
      Particle *_particles
    ) {
      unsigned int self = getLinearIndex();
      detectMinMax( x_boxel_count, y_boxel_count, z_boxel_count, minmax, boxel_size, _particles[ self ].position );
    }

    EPP_DEVICE void calcSize (
      SliceIndex *minmax,
      unsigned int _index
    ) {
      if( minmax[ _index ].x_max > minmax[ _index ].x_min ) {
        minmax[ _index ].width = minmax[ _index ].x_max - minmax[ _index ].x_min + 1;
        minmax[ _index ].temp_offset = minmax[ _index ].y_max - minmax[ _index ].y_min + 1;
        minmax[ _index ].temp_offset *= minmax[ _index ].width;
        minmax[ _index ].offset = 0;
      }
      else {
        minmax[ _index ].width = 0;
        minmax[ _index ].temp_offset = 0;
        minmax[ _index ].offset = 0;
      }
    }

    extern "C"
    EPP_GLOBAL void calcSizeAll(
      SliceIndex *minmax
    ) {
      unsigned int self = getLinearIndex();
      calcSize( minmax, self );
    }

    EPP_DEVICE void calcOffset (
      SliceIndex *slice_index,
      unsigned int _index,
      unsigned int *current_top
    ) {
      int sub_index;
      slice_index[ _index ].offset = atomicAdd( current_top, slice_index[ _index ].temp_offset );
    }

    extern "C"
    EPP_GLOBAL void calcOffsetAll(
      SliceIndex *minmax,
      unsigned int *current_top
    ) {
      unsigned int self = getLinearIndex();
      calcOffset( minmax, self, current_top );
    }

    /*
    EPP_DEVICE void calcOffsetPhase1 (
      SliceIndex *minmax,
      unsigned int _index,
      unsigned int _width
    ) {
      int sub_index;
      for( sub_index = 1; sub_index != _width; sub_index++ ) {
        minmax[ _index * _width + sub_index ].temp_offset +=
          minmax[ _index * _width + sub_index - 1 ].temp_offset;
      }
    }

    extern "C"
    EPP_GLOBAL void calcOffsetPhase1All(
      SliceIndex *minmax,
      unsigned int _width
    ) {
      unsigned int self = getLinearIndex();
      calcOffsetPhase1( minmax, self, _width );
    }

    EPP_DEVICE void calcOffsetPhase2 (
      SliceIndex *minmax,
      unsigned int _index,
      unsigned int _width,
      unsigned int _height
    ) {
      int sub_index;
      int offset = 0;
      for( sub_index = 1; sub_index != _height; sub_index++ ) {
        offset += minmax[ _width * sub_index - 1 ].temp_offset;
        minmax[ _width * sub_index + _index + 1 ].offset +=
          offset;
      }
    }

    extern "C"
    EPP_GLOBAL void calcOffsetPhase2All(
      SliceIndex *minmax,
      unsigned int _width,
      unsigned int _height
    ) {
      unsigned int self = getLinearIndex();
      calcOffsetPhase2( minmax, self, _width, _height );
    }
*/

    
    
    EPP_DEVICE void getRequiredSNSGSize(
      unsigned int *output,
      SliceIndex *minmax,
      unsigned int index
    ) {
      atomicMax( output, minmax[ index ].offset );
    }

    extern "C"
    EPP_GLOBAL void getRequiredSNSGSizeAll(
      unsigned int *output,
      SliceIndex *minmax
    ) {
      unsigned int self = getLinearIndex();
      getRequiredSNSGSize( output, minmax, self );
    }

    EPP_DEVICE void setSNSG(
      float boxel_size, unsigned int capacity,
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      unsigned int z_boxel_count,
      unsigned int *buffer,
      const float3 &_position, unsigned int _index,
      SliceIndex *slice_index
    ) {
      int3 position_in_grid = emath::vector_cast< int3 >( _position / boxel_size );
      position_in_grid.z = static_cast< unsigned int >( position_in_grid.z ) % z_boxel_count;
      position_in_grid.x -= slice_index[ position_in_grid.z ].x_min;
      position_in_grid.y -= slice_index[ position_in_grid.z ].y_min;
      position_in_grid.x = static_cast< unsigned int >( position_in_grid.x ) % x_boxel_count;
      position_in_grid.y = static_cast< unsigned int >( position_in_grid.y ) % y_boxel_count;
      int where_to_set = (
        slice_index[ position_in_grid.z ].offset +
        position_in_grid.y * slice_index[ position_in_grid.z ].width +
        position_in_grid.x ) * capacity;
      int element_index;
      _index++;
      for( element_index = 0; element_index != capacity; element_index++ )
        if( !atomicCAS( buffer + where_to_set + element_index, 0, _index ) )
          break;
    }

    extern "C"
    EPP_GLOBAL void setSNSGAll(
      float boxel_size, unsigned int capacity,
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      unsigned int z_boxel_count,
      unsigned int *buffer,
      Particle *_particles,
      SliceIndex *slice_index
    ) {
        unsigned int self = getLinearIndex();
        setSNSG( boxel_size, capacity, x_boxel_count, y_boxel_count, z_boxel_count, buffer, _particles[ self ].position, self, slice_index );
    }

    EPP_DEVICE void getSNSG(
      float boxel_size, unsigned int capacity,
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      unsigned int z_boxel_count,
      unsigned int *buffer,
      unsigned int *_output, unsigned int _self, const float3 &_position,
      SliceIndex *slice_index
    ) {
      int3 position_in_grid_original = emath::vector_cast< int3 >( _position / boxel_size );
      unsigned int offset_index;
      for( offset_index = 0; offset_index != 27; offset_index++ ) {
        int3 position_in_grid = position_in_grid_original + offsets[ offset_index ];
        position_in_grid.z = static_cast< unsigned int >( position_in_grid.z ) % z_boxel_count;
        position_in_grid.x -= slice_index[ position_in_grid.z ].x_min;
        position_in_grid.y -= slice_index[ position_in_grid.z ].y_min;
        position_in_grid.x = static_cast< unsigned int >( position_in_grid.x ) % x_boxel_count;
        position_in_grid.y = static_cast< unsigned int >( position_in_grid.y ) % y_boxel_count;
        if( slice_index[ position_in_grid.z ].width &&
            position_in_grid.x >= 0 &&
            position_in_grid.x < slice_index[ position_in_grid.z ].x_max - slice_index[ position_in_grid.z ].x_min &&
            position_in_grid.y >= 0 &&
            position_in_grid.y < slice_index[ position_in_grid.z ].y_max - slice_index[ position_in_grid.z ].y_min
          ) {
          int where_to_read = (
              slice_index[ position_in_grid.z ].offset +
              position_in_grid.y * slice_index[ position_in_grid.z ].width +
              position_in_grid.x ) * capacity;
          int element_index;
          for( element_index = 0; element_index != capacity - 1;  element_index++ ) {
            unsigned int value = buffer[ where_to_read + element_index ];
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
      }
      *_output = 0;
    }
  
    extern "C"
    EPP_GLOBAL void getSNSGAll(
      float boxel_size, unsigned int capacity,
      unsigned int x_boxel_count,
      unsigned int y_boxel_count,
      unsigned int z_boxel_count,
      unsigned int *buffer,
      unsigned int *_output, Particle *_particles,
      SliceIndex *slice_index
    ) {
      unsigned int self = getLinearIndex();
      getSNSG( boxel_size, capacity, x_boxel_count, y_boxel_count, z_boxel_count, buffer, _output + ( capacity * 27  * self ), self, _particles[ self ].position, slice_index );
    }
  }
#endif
}
#endif
