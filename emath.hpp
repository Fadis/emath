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

#ifndef EMATH_EMATH_HEADER
#define EMATH_EMATH_HEADER

#include <boost/preprocessor.hpp>
#include <boost/static_assert.hpp>

#include <emath/config.hpp>
#include <emath/types.hpp>
#include <emath/swap.hpp>
#include <emath/addsub.hpp>
#include <emath/scale.hpp>
#include <emath/dot.hpp>
#include <emath/length.hpp>
#include <emath/normalize.hpp>
#include <emath/volume.hpp>
#include <emath/clear.hpp>
#include <emath/cross.hpp>
#include <emath/outer.hpp>
#include <emath/exterior.hpp>
#include <emath/mult.hpp>
#include <emath/rot90.hpp>
#include <emath/voronoi.hpp>
#include <emath/distance.hpp>
#include <emath/memcpy.hpp>
#include <emath/det.hpp>
#include <emath/identity.hpp>
#include <emath/lu.hpp>
#include <emath/inverse.hpp>
#include <emath/dump.hpp>
#include <emath/cholesky.hpp>
#include <emath/serialize.hpp>
#include <emath/staticnsg.hpp>
#include <emath/multiarray.hpp>
#include <emath/mps.hpp>
#include <emath/slicensg.hpp>
#include <emath/treensg.hpp>
#include <emath/sin.hpp>

namespace emath {
}

#endif