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

#ifndef EMATH_TYPES_HEADER
#define EMATH_TYPES_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>

#define EPP_ELEMENT_NAME_LIST \
  ( 4, ( x, y, z, w ) )

#define EPP_VECTOR_TYPE_LIST \
  ( 11, ( \
    ( 4, ( char1, char2, char3, char4 ) ), \
    ( 4, ( uchar1, uchar2, uchar3, uchar4 ) ), \
    ( 4, ( short1, short2, short3, short4 ) ), \
    ( 4, ( ushort1, ushort2, ushort3, ushort4 ) ), \
    ( 4, ( int1, int2, int3, int4 ) ), \
    ( 4, ( uint1, uint2, uint3, uint4 ) ), \
    ( 4, ( long1, long2, long3, long4 ) ), \
    ( 4, ( ulong1, ulong2, ulong3, ulong4 ) ), \
    ( 4, ( longlong1, longlong2, longlong3, longlong4 ) ), \
    ( 4, ( float1, float2, float3, float4 ) ), \
    ( 4, ( double1, double2, double3, double4 ) ) \
  ) )

#define EPP_SCALAR_TYPE_LIST \
  ( 11, ( \
    signed char, \
    unsigned char, \
    signed short int, \
    unsigned short int, \
    signed int, \
    unsigned int, \
    signed long int, \
    unsigned long int, \
    signed long long int, \
    float, \
    double \
  ) )

#define EPP_ELEMENT( index ) \
  BOOST_PP_ARRAY_ELEM( index, EPP_ELEMENT_NAME_LIST )

#define EPP_MATRIX_ELEMENT( index, width ) \
  EPP_ELEMENT( BOOST_PP_DIV( index, width ) ).EPP_ELEMENT( BOOST_PP_MOD( index, width ) )

#define EPP_TRANSPOSED_MATRIX_ELEMENT( index, width ) \
  EPP_ELEMENT( BOOST_PP_MOD( index, width ) ).EPP_ELEMENT( BOOST_PP_DIV( index, width ) )

#define EPP_SCALAR_TYPE( type ) \
  BOOST_PP_ARRAY_ELEM( type, EPP_SCALAR_TYPE_LIST )

#define EPP_SCALER_TYPE_COUNT \
      BOOST_PP_ARRAY_SIZE ( EPP_SCALER_TYPE_LIST )

#define EPP_VECTOR_TYPE( type, size ) \
  BOOST_PP_ARRAY_ELEM( BOOST_PP_SUB( size, 1 ), BOOST_PP_ARRAY_ELEM( type, EPP_VECTOR_TYPE_LIST ) )

#define EPP_VECTOR_TYPE_COUNT \
      BOOST_PP_ARRAY_SIZE ( EPP_VECTOR_TYPE_LIST )

#define EPP_VECTOR_TYPE_MAX_SIZE( type ) \
  BOOST_PP_ADD( BOOST_PP_ARRAY_SIZE ( \
      BOOST_PP_ARRAY_ELEM ( type, EPP_VECTOR_TYPE_LIST ) \
    ), 1 )

#define EPP_VECTOR_TYPE_MIN_SIZE( type ) \
  1

#define EPP_MATRIX_TYPE( type, size ) \
  BOOST_PP_CAT( BOOST_PP_CAT( EPP_VECTOR_TYPE( type, size ), x ), size )

#define EPP_MATRIX_TYPE_COUNT \
      BOOST_PP_ARRAY_SIZE ( EPP_VECTOR_TYPE_LIST )

#define EPP_MATRIX_TYPE_MAX_SIZE( type ) \
  BOOST_PP_ADD( BOOST_PP_ARRAY_SIZE ( \
      BOOST_PP_ARRAY_ELEM ( type, EPP_VECTOR_TYPE_LIST ) \
    ), 1 )

#define EPP_MATRIX_TYPE_MIN_SIZE( type ) \
  1

#define EPP_MEMBER( z, index, type ) \
  type EPP_ELEMENT( index );

#ifndef __CUDACC__
#include <eagle/cuda/cuda.hpp>
/*
#define EPP_VECTOR_STRUCT( z, index, type ) \
  struct EPP_VECTOR_TYPE( type, index ) { \
    BOOST_PP_REPEAT( \
      index, \
      EPP_MEMBER, \
      EPP_SCALAR_TYPE( type ) \
    ) \
  };

#define EPP_VECTOR( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_VECTOR_STRUCT, \
    type \
  )

BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_VECTOR, EPP_UNUSED )
*/
  struct longlong3 {
  long long int x;
  long long int y;
  long long int z;
  };
  struct longlong4 {
    long long int x;
    long long int y;
    long long int z;
    long long int w;
  };
  struct double3 {
    double x;
    double y;
    double z;
  };
  struct double4 {
    double x;
    double y;
    double z;
    double w;
  };

#else
  struct longlong3 {
    long long int x;
    long long int y;
    long long int z;
  };
  struct longlong4 {
    long long int x;
    long long int y;
    long long int z;
    long long int w;
  };
  struct double3 {
    double x;
    double y;
    double z;
  };
  struct double4 {
    double x;
    double y;
    double z;
    double w;
  };
#endif

  #define EPP_MATRIX_STRUCT( z, index, type ) \
  struct EPP_MATRIX_TYPE( type, index ) { \
    BOOST_PP_REPEAT( \
      index, \
      EPP_MEMBER, \
      EPP_VECTOR_TYPE( type, index ) \
    ) \
  };

#define EPP_MATRIX( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_MATRIX_STRUCT, \
    type \
  )

BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_MATRIX, EPP_UNUSED )

namespace emath {
  template< typename VectorType >
    struct Traits {
    };

#define EPP_TEMPLATE_TYPE_OFFSETS( z, index, type ) \
   & type :: EPP_ELEMENT( index ),

#define EPP_TEMPLATE_TYPE_STRUCT( z, index, type ) \
  template<> \
    struct Traits< EPP_VECTOR_TYPE( type, index ) > { \
      static const unsigned int element_count = index;\
      typedef EPP_SCALAR_TYPE( type ) ElementType; \
      static EPP_SCALAR_TYPE( type ) EPP_VECTOR_TYPE( type, index ) :: * const offsets[ index ]; \
    }; \
  template<> \
    struct Traits< EPP_MATRIX_TYPE( type, index ) > { \
      static const unsigned int element_count = index;\
      typedef EPP_VECTOR_TYPE( type, index ) ElementType; \
      static EPP_VECTOR_TYPE( type, index ) EPP_MATRIX_TYPE( type, index ) :: * const offsets[ index ]; \
    }; \
  template<> \
    struct Traits< const EPP_VECTOR_TYPE( type, index ) > { \
      static const unsigned int element_count = index;\
      typedef const EPP_SCALAR_TYPE( type ) ElementType; \
      static const EPP_SCALAR_TYPE( type ) EPP_VECTOR_TYPE( type, index ) :: * const offsets[ index ]; \
    }; \
  template<> \
    struct Traits< const EPP_MATRIX_TYPE( type, index ) > { \
      static const unsigned int element_count = index;\
      typedef const EPP_VECTOR_TYPE( type, index ) ElementType; \
      static const EPP_VECTOR_TYPE( type, index ) EPP_MATRIX_TYPE( type, index ) :: * const offsets[ index ]; \
    };

#define EPP_TEMPLATE_TYPE_DATA( z, index, type ) \
    EPP_SCALAR_TYPE( type ) EPP_VECTOR_TYPE( type, index ) :: * const Traits< EPP_VECTOR_TYPE( type, index ) > :: offsets[ index ] = { \
      BOOST_PP_REPEAT( index, EPP_TEMPLATE_TYPE_OFFSETS, EPP_VECTOR_TYPE( type, index ) ) \
    }; \
    EPP_VECTOR_TYPE( type, index ) EPP_MATRIX_TYPE( type, index ) :: * const Traits< EPP_MATRIX_TYPE( type, index ) > :: offsets[ index ] = { \
      BOOST_PP_REPEAT( index, EPP_TEMPLATE_TYPE_OFFSETS, EPP_MATRIX_TYPE( type, index ) ) \
    }; \
    const EPP_SCALAR_TYPE( type ) EPP_VECTOR_TYPE( type, index ) :: * const Traits< const EPP_VECTOR_TYPE( type, index ) > :: offsets[ index ] = { \
      BOOST_PP_REPEAT( index, EPP_TEMPLATE_TYPE_OFFSETS, EPP_VECTOR_TYPE( type, index ) ) \
    }; \
    const EPP_VECTOR_TYPE( type, index ) EPP_MATRIX_TYPE( type, index ) :: * const Traits< const EPP_MATRIX_TYPE( type, index ) > ::  offsets[ index ] = { \
      BOOST_PP_REPEAT( index, EPP_TEMPLATE_TYPE_OFFSETS, EPP_MATRIX_TYPE( type, index ) ) \
    };

#define EPP_TEMPLATE_TYPE( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_TEMPLATE_TYPE_STRUCT, \
    type \
  )

#define EPP_TEMPLATE_TYPE_DATAS( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_TEMPLATE_TYPE_DATA, \
    type \
  )

BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_TEMPLATE_TYPE, EPP_UNUSED )

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_TEMPLATE_TYPE_DATAS, EPP_UNUSED )
#endif

#define EPP_TEMPLATE_GET_FUNCTION( z, index, type ) \
  EPP_SCALAR_TYPE( type ) & get( EPP_VECTOR_TYPE( type, index ) &_source, unsigned int _index ) { \
    return _source.*Traits< EPP_VECTOR_TYPE( type, index ) >::offsets[ _index ]; \
  } \
  EPP_SCALAR_TYPE( type ) & get( EPP_MATRIX_TYPE( type, index ) &_source, unsigned int _row, unsigned int _col ) { \
  return _source.*Traits< EPP_MATRIX_TYPE( type, index ) >::offsets[ _row ].*Traits< EPP_VECTOR_TYPE( type, index ) >::offsets[ _col ]; \
  } \
  const EPP_SCALAR_TYPE( type ) & get( const EPP_VECTOR_TYPE( type, index ) &_source, unsigned int _index ) { \
    return _source.*Traits< const EPP_VECTOR_TYPE( type, index ) >::offsets[ _index ]; \
  } \
  const EPP_SCALAR_TYPE( type ) & get( const EPP_MATRIX_TYPE( type, index ) &_source, unsigned int _row, unsigned int _col ) { \
  return _source.*Traits< const EPP_MATRIX_TYPE( type, index ) >::offsets[ _row ].*Traits< const EPP_VECTOR_TYPE( type, index ) >::offsets[ _col ]; \
  }

#define EPP_TEMPLATE_GET_PROTOTYPE( z, index, type ) \
  EPP_SCALAR_TYPE( type ) & get( EPP_VECTOR_TYPE( type, index ) &_source, unsigned int _index ); \
  EPP_SCALAR_TYPE( type ) & get( EPP_MATRIX_TYPE( type, index ) &_source, unsigned int _row, unsigned int _col ); \
  const EPP_SCALAR_TYPE( type ) & get( const EPP_VECTOR_TYPE( type, index ) &_source, unsigned int _index ); \
  const EPP_SCALAR_TYPE( type ) & get( const EPP_MATRIX_TYPE( type, index ) &_source, unsigned int _row, unsigned int _col );

#define EPP_TEMPLATE_GET( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_TEMPLATE_GET_FUNCTION, \
    type \
  )

#define EPP_TEMPLATE_GET_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_TEMPLATE_GET_PROTOTYPE, \
    type \
  )

#ifdef __CUDACC__
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_TEMPLATE_GET, EPP_UNUSED )
#else
BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_TEMPLATE_GET_PROTOTYPES, EPP_UNUSED )
#endif

  template< unsigned int left, unsigned int right >
    class Min {
      public:
        static const unsigned int value = ( left < right ) ? left : right;
    };

  template< typename ToType, typename FromType,
    int size = Min< Traits< ToType >::element_count, Traits< FromType >::element_count >::value >
    class VectorCastCore {};

  template< typename ToType, typename FromType >
    class VectorCastCore< ToType, FromType, 1 > {
      public:
        EPP_DEVICE static void run( ToType &_dest, FromType &_source ) {
          _dest.x = _source.x;
        }
    };

  template< typename ToType, typename FromType >
    class VectorCastCore< ToType, FromType, 2 > {
      public:
        EPP_DEVICE static void run( ToType &_dest, FromType &_source ) {
          _dest.x = _source.x;
          _dest.y = _source.y;
        }
    };

  template< typename ToType, typename FromType >
    class VectorCastCore< ToType, FromType, 3 > {
      public:
        EPP_DEVICE static void run( ToType &_dest, FromType &_source ) {
          _dest.x = _source.x;
          _dest.y = _source.y;
          _dest.z = _source.z;
        }
    };

  template< typename ToType, typename FromType >
    class VectorCastCore< ToType, FromType, 4 > {
      public:
        EPP_DEVICE static void run( ToType &_dest, FromType &_source ) {
          _dest.x = _source.x;
          _dest.y = _source.y;
          _dest.z = _source.z;
          _dest.w = _source.w;
        }
    };


  template< typename ToType, typename FromType >
    EPP_DEVICE ToType vector_cast( FromType _value ) {
      ToType temp;
      VectorCastCore< ToType, FromType >::run( temp, _value );
      return temp;
    }
}

#endif
