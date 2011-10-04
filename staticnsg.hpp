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

#ifndef ENSG_STATICNSG_HEADER
#define ENSG_STATICNSG_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/scale.hpp>
#include <emath/multiarray.hpp>
#include <emath/mps.hpp>

namespace emath {
#ifdef __CUDACC__

  EPP_DEVICE unsigned int getLinearIndex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }
  /*
  EPP_DEVICE void clearNSG(
    unsigned int capacity,
    unsigned int *buffer,
    unsigned int _index
  ) {
    int counter;
    for( counter = 0; counter != capacity; counter++ )
      buffer[ capacity * _index + counter ] = 0;
  }

  extern "C"
  EPP_GLOBAL void clearNSGAll(
    unsigned int capacity,
    unsigned int *buffer
  ) {
    unsigned int self = getLinearIndex();
    clearNSG( capacity, buffer, self );
  }

  */

  EPP_DEVICE void clearNSG(
    unsigned int capacity,
    unsigned int *buffer,
    unsigned int _index,
    unsigned int _cycle
  ) {
    int counter;
    for( counter = 0; counter != capacity * _cycle; counter++ )
      buffer[ capacity * _cycle * _index + counter ] = 0;
  }

  extern "C"
  EPP_GLOBAL void clearNSGAll(
    unsigned int capacity,
    unsigned int *buffer,
    unsigned int _cycle
  ) {
    unsigned int self = getLinearIndex();
    clearNSG( capacity, buffer, self, _cycle );
  }

  EPP_DEVICE void clearNSGOutput(
    unsigned int capacity,
    unsigned int *output,
    unsigned int _index
  ) {
    int counter;
    for( counter = 0; counter != 27 * capacity; counter++ )
      output[ capacity * 27 * _index + counter ] = 0;
  }

  extern "C"
  EPP_GLOBAL void clearNSGOutputAll(
    unsigned int capacity,
    unsigned int *output
  ) {
    unsigned int self = getLinearIndex();
    clearNSGOutput( capacity, output, self );
  }
  
  EPP_DEVICE uint3 hash( uint3 size, int3 value ) {
    uint3 temp;
    int3 scaled_value = value;
    temp.x = scaled_value.x % size.x;
    temp.y = scaled_value.y % size.y;
    temp.z = scaled_value.z % size.z;
    return temp;
  }

  EPP_DEVICE void setNSG(
    float boxel_size, unsigned int capacity,
    unsigned int x_boxel_count,
    unsigned int y_boxel_count,
    unsigned int z_boxel_count,
    unsigned int *buffer,
    const float3 &_position, unsigned int _index
  ) {
    int3 position_in_grid = emath::vector_cast< int3 >( _position / boxel_size );
//    int3 position_in_grid;
//    position_in_grid.x = _position.x / boxel_size;
//    position_in_grid.y = _position.y / boxel_size;
//    position_in_grid.z = _position.z / boxel_size;
    position_in_grid.x %= x_boxel_count;
    position_in_grid.y %= y_boxel_count;
    position_in_grid.z %= z_boxel_count;
    unsigned int where_to_set =
        ( position_in_grid.x +
        position_in_grid.y * x_boxel_count +
        position_in_grid.z * x_boxel_count * y_boxel_count ) * capacity;
    unsigned int element_index;
    _index++;
    for( element_index = 0; element_index != capacity; element_index++ )
      if( !atomicCAS( buffer + where_to_set + element_index, 0, _index ) )
        break;
  }

  extern "C"
  EPP_GLOBAL void setNSGAll(
    float boxel_size, unsigned int capacity,
    unsigned int x_boxel_count,
    unsigned int y_boxel_count,
    unsigned int z_boxel_count,
    unsigned int *buffer,
    Particle *_particles
  ) {
      unsigned int self = getLinearIndex();
      setNSG( boxel_size, capacity, x_boxel_count, y_boxel_count, z_boxel_count, buffer, _particles[ self ].position, self );
  }


  EPP_DEVICE void getNSG(
    float boxel_size, unsigned int capacity,
    unsigned int x_boxel_count,
    unsigned int y_boxel_count,
    unsigned int z_boxel_count,
    unsigned int *buffer,
    unsigned int *_output, unsigned int _self, const float3 &_position
  ) {
    int3 position_in_grid_original = emath::vector_cast< int3 >( _position / boxel_size );
    unsigned int offset_index;
    int3 counts = make_int3( x_boxel_count, y_boxel_count, z_boxel_count );
    for( offset_index = 0; offset_index != 27; offset_index++ ) {
      int3 position_in_grid = position_in_grid_original + counts + offsets[ offset_index ];
      position_in_grid.x %= x_boxel_count;
      position_in_grid.y %= y_boxel_count;
      position_in_grid.z %= z_boxel_count;
      unsigned int where_to_read =
          ( position_in_grid.x +
          position_in_grid.y * x_boxel_count +
          position_in_grid.z * x_boxel_count * y_boxel_count ) * capacity;
      unsigned int element_index;
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
    *_output = 0;
  }

  extern "C"
  EPP_GLOBAL void getNSGAll(
    float boxel_size, unsigned int capacity,
    unsigned int x_boxel_count,
    unsigned int y_boxel_count,
    unsigned int z_boxel_count,
    unsigned int *buffer,
    unsigned int *_output, Particle *_particles
  ) {
    unsigned int self = getLinearIndex();
    getNSG( boxel_size, capacity, x_boxel_count, y_boxel_count, z_boxel_count, buffer, _output + ( capacity * 27  * self ), self, _particles[ self ].position );
  }
  
  EPP_DEVICE Particle &getParticle(
    Particle *_particles,
    unsigned int *_neighbors,
    unsigned int _index
  ) {
    return _particles[ _neighbors[ _index ] - 1 ];
  }
/*
  class StaticNSG {
    public:
      EPP_DEVICE StaticNSG( float _boxel_size, unsigned int _capacity,
                            unsigned int _x_boxel_count,
                            unsigned int _y_boxel_count,
                            unsigned int _z_boxel_count,
                            unsigned int *_buffer )
      : boxel_size( _boxel_size ), capacity( _capacity ),
        x_boxel_count( _x_boxel_count ),
        y_boxel_count( _y_boxel_count ),
        z_boxel_count( _z_boxel_count ),
        buffer( _buffer ) {}
      EPP_DEVICE void set( const float3 &_position, unsigned int _index ) {
        int3 position_in_grid = emath::vector_cast< int3 >( _position / boxel_size );
        position_in_grid.x %= x_boxel_count;
        position_in_grid.y %= y_boxel_count;
        position_in_grid.z %= z_boxel_count;
        unsigned int where_to_set =
          ( position_in_grid.x +
            position_in_grid.y * x_boxel_count +
            position_in_grid.z * x_boxel_count * y_boxel_count ) * capacity;
        unsigned int element_index;
        for( element_index = 0; element_index != capacity; element_index++ )
          if( !atomicCAS( buffer + where_to_set + element_index, 0, _index ) )
            break;
      }
      EPP_DEVICE void get( unsigned int *_output, unsigned int _self, const float3 &_position ) {
        int3 position_in_grid_original = emath::vector_cast< int3 >( _position / boxel_size );
        unsigned int offset_index;
        int3 counts = make_int3( x_boxel_count, y_boxel_count, z_boxel_count );
        for( offset_index = 0; offset_index != 27; offset_index++ ) {
          int3 position_in_grid = position_in_grid_original + counts + offsets[ offset_index ];
          position_in_grid.x %= x_boxel_count;
          position_in_grid.y %= y_boxel_count;
          position_in_grid.z %= z_boxel_count;
          unsigned int where_to_read =
            ( position_in_grid.x +
              position_in_grid.y * x_boxel_count +
              position_in_grid.z * x_boxel_count * y_boxel_count ) * capacity;
          unsigned int element_index;
          for( element_index = 0; element_index != capacity; element_index++ )
            if( buffer[ where_to_read + element_index ] && buffer[ where_to_read + element_index ] != _self ) {
              *_output = buffer[ where_to_read + element_index ];
              _output++;
            }
        }
        *_output = 0;
      }
    private:
      float boxel_size;
      unsigned int capacity;
      unsigned int x_boxel_count;
      unsigned int y_boxel_count;
      unsigned int z_boxel_count;
      unsigned int *buffer;
  };
  */
#endif
}

#endif
