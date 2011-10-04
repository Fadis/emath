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

#ifndef EMATH_MPS_HEADER
#define EMATH_MPS_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#include <emath/addsub.hpp>
#include <emath/scale.hpp>
#include <emath/distance.hpp>
#include <emath/dump.hpp>

namespace emath {
  EPP_DEVICE const int3 offsets[] = {
    {  0,  0,  0, },
    {  1,  0,  0, },
    {  0,  1,  0, },
    {  0,  0,  1, },
    { -1,  0,  0, },
    {  0, -1,  0, },
    {  0,  0, -1, },
    {  0,  1,  1, },
    {  1,  0,  1, },
    {  1,  1,  0, },
    {  0, -1,  1, },
    {  1,  0, -1, },
    { -1,  1,  0, },
    {  0,  1, -1, },
    { -1,  0,  1, },
    {  1, -1,  0, },
    {  0, -1, -1, },
    { -1,  0, -1, },
    { -1, -1,  0, },
    { -1,  1,  1, },
    {  1, -1,  1, },
    {  1,  1, -1, },
    { -1, -1,  1, },
    {  1, -1, -1, },
    { -1,  1, -1, },
    {  1,  1,  1, },
    { -1, -1, -1, },
  };

  struct Particle {
    float3 position;
    float3 velocity;
    float pressure;
    float pressure_dd;
    float current_particle_density;
    float debug[ 8 ];
    float current_curvature;
    float temporary;
  };

  struct FluidSpecification {
    float particle_density; // n0
    float density; // ρ
    float viscosity; // μ
    float gravity; // g
    float kernel_size;
    unsigned int particle_count;
    float lambda;
    float delta;
    unsigned int pressure_cycle;
    float default_curvature;
    float surface_tension;
  };

  struct SliceIndex {
    int x_min;
    int y_min;
    int x_max;
    int y_max;
    unsigned int width;
    unsigned int offset;
    unsigned int temp_offset;
  };

  struct SliceLocationCache {
    int3 pos;
  };

  struct TreeLocationCache {
    uint3 pos;
    unsigned int primary_linear_pos;
    unsigned int secondary_linear_pos;
  };

  template< typename Type >
    EPP_DEVICE Type mpsKernel( Type range, Type pos ) {
    Type result = static_cast< Type >( 0.0 );
      if( pos >= static_cast< Type >( 1.0e-10 ) )
        result = ( range / pos ) - static_cast< Type >( 1.0 );
      if( result < static_cast< Type >( 0.0 ) )
        result = static_cast< Type >( 0.0 );
      return result;
    }

  template< typename Type >
    EPP_DEVICE Type getLambdaLowerOnAxis( Type _area, Type _interval ) {
      std::pair< Type, Type > current_pos;
      Type sum = static_cast< Type >( 0.0 );
      current_pos.second = static_cast< Type >( 0.0 );
      for( current_pos.first = static_cast< Type >( 0.0 ); current_pos.first < _area; current_pos.first += _interval ) {
        Type length = current_pos.first;
        if( length != static_cast< Type >( 0.0 ) ) {
          sum += mpsKernel( _area, length );
        }
      }
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getLambdaLowerOnPlane( Type _area, Type _interval, Type _shift ) {
      std::pair< Type, Type > current_pos;
      Type sum = static_cast< Type >( 0.0 );
      for( current_pos.second = static_cast< Type >( 0.0 ); current_pos.second < _area; current_pos.second += _interval ) {
        for( current_pos.first = static_cast< Type >( 0.0 ); current_pos.first < _area; current_pos.first += _interval ) {
          Type length = sqrt( current_pos.first * current_pos.first + current_pos.second * current_pos.second + _shift * _shift );
          if( length != static_cast< Type >( 0.0 ) ) {
            sum += mpsKernel( _area, length );
#ifndef __CUDACC__
            std::cout << "l( " << current_pos.first << ", " << current_pos.second << ", " << _shift << " ) ";
            std::cout << sum << std::endl;
#endif
          }
        }
      }
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getLambdaLower( Type _area, Type _interval ) {
      Type border = getLambdaLowerOnPlane( _area, _interval, static_cast< Type >( 0.0 ) );
      Type axis = getLambdaLowerOnAxis( _area, _interval );
      Type sum = static_cast< Type >( 0.0 );
      Type shift;
      for( shift = static_cast< Type >( 0.0 ); shift < _area; shift += _interval ) {
        sum += getLambdaLowerOnPlane( _area, _interval, shift );
      }
      sum *= 8;
      sum -= 4 * 3 * border;
      sum += 6 * axis;
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getLambdaUpperOnAxis( Type _area, Type _interval ) {
      std::pair< Type, Type > current_pos;
      Type sum = static_cast< Type >( 0.0 );
      current_pos.second = static_cast< Type >( 0.0 );
      for( current_pos.first = static_cast< Type >( 0.0 ); current_pos.first < _area; current_pos.first += _interval ) {
        Type length = current_pos.first;
        if( length != static_cast< Type >( 0.0 ) ) {
          sum += mpsKernel( _area, length ) * length * length;
        }
      }
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getLambdaUpperOnPlane( Type _area, Type _interval, Type _shift ) {
      std::pair< Type, Type > current_pos;
      Type sum = static_cast< Type >( 0.0 );
      for( current_pos.second = static_cast< Type >( 0.0 ); current_pos.second < _area; current_pos.second += _interval ) {
        for( current_pos.first = static_cast< Type >( 0.0 ); current_pos.first < _area; current_pos.first += _interval ) {
          Type length = sqrt( current_pos.first * current_pos.first + current_pos.second * current_pos.second + _shift * _shift );
          if( length != static_cast< Type >( 0.0 ) ) {
            sum += mpsKernel( _area, length ) * length * length;
#ifndef __CUDACC__
            std::cout << "u( " << current_pos.first << ", " << current_pos.second << ", " << _shift << " ) ";
            std::cout << sum << std::endl;
#endif
          }
        }
      }
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getLambdaUpper( Type _area, Type _interval ) {
      Type border = getLambdaUpperOnPlane( _area, _interval, static_cast< Type >( 0.0 ) );
      Type axis = getLambdaUpperOnAxis( _area, _interval );
      Type sum = static_cast< Type >( 0.0 );
      Type shift;
      for( shift = static_cast< Type >( 0.0 ); shift < _area; shift += _interval ) {
        sum += getLambdaUpperOnPlane( _area, _interval, shift );
      }
      sum *= 8;
      sum -= 4 * 3 * border;
      sum += 6 * axis;
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getLambda( Type _area, Type _interval ) {
      return getLambdaUpper( _area, _interval ) / getLambdaLower( _area, _interval );
    }
    
  template< typename Type >
    EPP_DEVICE Type mpsCurvatureKernel( Type range, Type pos ) {
      Type result = static_cast< Type >( 0.0 );
      if( pos <= range )
        result = static_cast< Type >( 1.0 );
      return result;
    }

  template< typename Type >
    EPP_DEVICE Type getDefaultCurvatureOnAxis( Type _area, Type _interval ) {
      std::pair< Type, Type > current_pos;
      Type sum = static_cast< Type >( 0.0 );
      current_pos.second = static_cast< Type >( 0.0 );
      for( current_pos.first = static_cast< Type >( 0.0 ); current_pos.first < _area; current_pos.first += _interval ) {
        Type length = current_pos.first;
        if( length != static_cast< Type >( 0.0 ) ) {
          sum += mpsCurvatureKernel( _area, length );
        }
      }
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getDefaultCurvatureOnPlane( Type _area, Type _interval, Type _shift ) {
      std::pair< Type, Type > current_pos;
      Type sum = static_cast< Type >( 0.0 );
      for( current_pos.second = static_cast< Type >( 0.0 ); current_pos.second < _area; current_pos.second += _interval ) {
        for( current_pos.first = static_cast< Type >( 0.0 ); current_pos.first < _area; current_pos.first += _interval ) {
          Type length = sqrt( current_pos.first * current_pos.first + current_pos.second * current_pos.second + _shift * _shift );
          if( length != static_cast< Type >( 0.0 ) ) {
            sum += mpsCurvatureKernel( _area, length );
#ifndef __CUDACC__
            std::cout << "c( " << current_pos.first << ", " << current_pos.second << ", " << _shift << " ) ";
            std::cout << sum << std::endl;
#endif
          }
        }
      }
      return sum;
    }

  template< typename Type >
    EPP_DEVICE Type getDefaultCurvature( Type _area, Type _interval ) {
      Type border = getDefaultCurvatureOnPlane( _area, _interval, static_cast< Type >( 0.0 ) );
      Type axis = getDefaultCurvatureOnAxis( _area, _interval );
      Type sum = static_cast< Type >( 0.0 );
      Type shift;
      for( shift = static_cast< Type >( 0.0 ); shift < _area; shift += _interval ) {
        sum += getDefaultCurvatureOnPlane( _area, _interval, shift );
      }
      sum *= 4;
      sum -= 4 * border;
      sum += axis;
      return sum;
    }
}

#endif
