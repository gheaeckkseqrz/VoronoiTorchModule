require 'cutorch'
require 'image'
voronoi = require 'voronoi'

local function main()

   math.randomseed(11)

   local map = torch.zeros(3, 1024, 1024):cuda()
   local seeds = {}
   for i=1, 50 do
      table.insert(seeds, {{math.random(1, map:size(3)), math.random(1, map:size(2))}, {math.random(100, 255), math.random(100, 255), math.random(100, 255)}})
   end

--   while true do
      map:zero()
      for i=1, #seeds do

	 seeds[i][1][1] = math.max(1, math.min(seeds[i][1][1] + math.random(-1, 1), map:size(3))) -- Seed X
	 seeds[i][1][2] = math.max(1, math.min(seeds[i][1][2] + math.random(-1, 1), map:size(2))) -- Seed Y
	 local pix = map:narrow(3, seeds[i][1][1], 1):narrow(2, seeds[i][1][2], 1)
	 pix[1] = seeds[i][2][1]
	 pix[2] = seeds[i][2][2]
	 pix[3] = seeds[i][2][3]
      end
      win1 = image.display{image=map, win=win1}
      voronoi.computeVoronoi(map)
      win2 = image.display{image=map, win=win2}

      image.save('voronoi.png', map:div(255))


  -- end
end

main()
