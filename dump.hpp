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

#ifndef EMATH_DUMP_HEADER
#define EMATH_DUMP_HEADER

#ifdef __CUDACC__
#warning Dumper is not available for CUDA.
#else

#include <iostream>

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>

#define EPP_DUMP_VECTOR_OPERATION( z, index, unused ) \
  _stream << _source.EPP_ELEMENT( index ) << ", ";

#define EPP_DUMP_VECTOR_FUNCTION( z, index, type ) \
  std::ostream &operator<< ( \
    std::ostream& _stream, \
    const EPP_VECTOR_TYPE( type, index ) &_source \
  ) { \
    _stream << "( "; \
    BOOST_PP_REPEAT( \
      index, \
      EPP_DUMP_VECTOR_OPERATION, \
      EPP_UNUSED \
    ) \
    _stream << ")"; \
    return _stream; \
  }

#define EPP_DUMP_VECTOR_PROTOTYPE( z, index, type ) \
  std::ostream &operator<< ( \
    std::ostream& _stream, \
    const EPP_VECTOR_TYPE( type, index ) &_source \
  );

#define EPP_DUMP_VECTOR( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DUMP_VECTOR_FUNCTION, \
    type \
  )

#define EPP_DUMP_VECTOR_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_VECTOR_TYPE_MIN_SIZE( type ), \
    EPP_VECTOR_TYPE_MAX_SIZE( type ), \
    EPP_DUMP_VECTOR_PROTOTYPE, \
    type \
  )


#define EPP_DUMP_MATRIX_OPERATION( z, index, width ) \
  _stream << _source.EPP_MATRIX_ELEMENT( index, width ) << ", ";

#define EPP_DUMP_MATRIX_FUNCTION( z, index, type ) \
  std::ostream &operator<< ( \
    std::ostream& _stream, \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  ) { \
    _stream << "( "; \
    BOOST_PP_REPEAT( \
      BOOST_PP_MUL( index, index ), \
      EPP_DUMP_MATRIX_OPERATION, \
      index \
    ) \
    _stream << ")"; \
    return _stream; \
  }

#define EPP_DUMP_MATRIX_PROTOTYPE( z, index, type ) \
  std::ostream &operator<< ( \
    std::ostream& _stream, \
    const EPP_MATRIX_TYPE( type, index ) &_source \
  );

#define EPP_DUMP_MATRIX( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_DUMP_MATRIX_FUNCTION, \
    type \
  )

#define EPP_DUMP_MATRIX_PROTOTYPES( z, type, unused ) \
  BOOST_PP_REPEAT_FROM_TO ( \
    EPP_MATRIX_TYPE_MIN_SIZE( type ), \
    EPP_MATRIX_TYPE_MAX_SIZE( type ), \
    EPP_DUMP_MATRIX_PROTOTYPE, \
    type \
  )
//namespace emath {

BOOST_PP_REPEAT( EPP_VECTOR_TYPE_COUNT, EPP_DUMP_VECTOR_PROTOTYPES, EPP_UNUSED )
BOOST_PP_REPEAT( EPP_MATRIX_TYPE_COUNT, EPP_DUMP_MATRIX_PROTOTYPES, EPP_UNUSED )

//}

#endif

#endif
