local voronoi = {}

voronoiinit = false

function voronoi.computeVoronoi(input)
   if voronoiinit == false then
      ffi = require 'ffi'
      ffi.cdef[[ void computeVoronoi(float *input, int x, int y); ]]
      local libFile = debug.getinfo(1).source:match("@(.*)$"):gsub('voronoi.lua', 'libvoronoi.so') -- Quite hacky :/
      clib = ffi.load(libFile)
      voronoiinit = true
   end
   clib.computeVoronoi(input:data(), input:size(2), input:size(3))
end

return voronoi
