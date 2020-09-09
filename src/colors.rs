use phf::{phf_map, phf_set};

pub(crate) static COLORS: phf::Map<&'static str, [u8; 4]> = phf_map! {
    "snow" => [255, 250, 250, 0xff],
"ghost white" => [248, 248, 255, 0xff],
"GhostWhite" => [248, 248, 255, 0xff],
"white smoke" => [245, 245, 245, 0xff],
"WhiteSmoke" => [245, 245, 245, 0xff],
"gainsboro" => [220, 220, 220, 0xff],
"floral white" => [255, 250, 240, 0xff],
"FloralWhite" => [255, 250, 240, 0xff],
"old lace" => [253, 245, 230, 0xff],
"OldLace" => [253, 245, 230, 0xff],
"linen" => [250, 240, 230, 0xff],
"antique white" => [250, 235, 215, 0xff],
"AntiqueWhite" => [250, 235, 215, 0xff],
"papaya whip" => [255, 239, 213, 0xff],
"PapayaWhip" => [255, 239, 213, 0xff],
"blanched almond" => [255, 235, 205, 0xff],
"BlanchedAlmond" => [255, 235, 205, 0xff],
"bisque" => [255, 228, 196, 0xff],
"peach puff" => [255, 218, 185, 0xff],
"PeachPuff" => [255, 218, 185, 0xff],
"navajo white" => [255, 222, 173, 0xff],
"NavajoWhite" => [255, 222, 173, 0xff],
"moccasin" => [255, 228, 181, 0xff],
"cornsilk" => [255, 248, 220, 0xff],
"ivory" => [255, 255, 240, 0xff],
"lemon chiffon" => [255, 250, 205, 0xff],
"LemonChiffon" => [255, 250, 205, 0xff],
"seashell" => [255, 245, 238, 0xff],
"honeydew" => [240, 255, 240, 0xff],
"mint cream" => [245, 255, 250, 0xff],
"MintCream" => [245, 255, 250, 0xff],
"azure" => [240, 255, 255, 0xff],
"alice blue" => [240, 248, 255, 0xff],
"AliceBlue" => [240, 248, 255, 0xff],
"lavender" => [230, 230, 250, 0xff],
"lavender blush" => [255, 240, 245, 0xff],
"LavenderBlush" => [255, 240, 245, 0xff],
"misty rose" => [255, 228, 225, 0xff],
"MistyRose" => [255, 228, 225, 0xff],
"white" => [255, 255, 255, 0xff],
"black" => [0, 0, 0, 0xff],
"dark slate gray" => [47, 79, 79, 0xff],
"DarkSlateGray" => [47, 79, 79, 0xff],
"dark slate grey" => [47, 79, 79, 0xff],
"DarkSlateGrey" => [47, 79, 79, 0xff],
"dim gray" => [105, 105, 105, 0xff],
"DimGray" => [105, 105, 105, 0xff],
"dim grey" => [105, 105, 105, 0xff],
"DimGrey" => [105, 105, 105, 0xff],
"slate gray" => [112, 128, 144, 0xff],
"SlateGray" => [112, 128, 144, 0xff],
"slate grey" => [112, 128, 144, 0xff],
"SlateGrey" => [112, 128, 144, 0xff],
"light slate gray" => [119, 136, 153, 0xff],
"LightSlateGray" => [119, 136, 153, 0xff],
"light slate grey" => [119, 136, 153, 0xff],
"LightSlateGrey" => [119, 136, 153, 0xff],
"gray" => [190, 190, 190, 0xff],
"grey" => [190, 190, 190, 0xff],
"x11 gray" => [190, 190, 190, 0xff],
"X11Gray" => [190, 190, 190, 0xff],
"x11 grey" => [190, 190, 190, 0xff],
"X11Grey" => [190, 190, 190, 0xff],
"web gray" => [128, 128, 128, 0xff],
"WebGray" => [128, 128, 128, 0xff],
"web grey" => [128, 128, 128, 0xff],
"WebGrey" => [128, 128, 128, 0xff],
"light grey" => [211, 211, 211, 0xff],
"LightGrey" => [211, 211, 211, 0xff],
"light gray" => [211, 211, 211, 0xff],
"LightGray" => [211, 211, 211, 0xff],
"midnight blue" => [25, 25, 112, 0xff],
"MidnightBlue" => [25, 25, 112, 0xff],
"navy" => [0, 0, 128, 0xff],
"navy blue" => [0, 0, 128, 0xff],
"NavyBlue" => [0, 0, 128, 0xff],
"cornflower blue" => [100, 149, 237, 0xff],
"CornflowerBlue" => [100, 149, 237, 0xff],
"dark slate blue" => [72, 61, 139, 0xff],
"DarkSlateBlue" => [72, 61, 139, 0xff],
"slate blue" => [106, 90, 205, 0xff],
"SlateBlue" => [106, 90, 205, 0xff],
"medium slate blue" => [123, 104, 238, 0xff],
"MediumSlateBlue" => [123, 104, 238, 0xff],
"light slate blue" => [132, 112, 255, 0xff],
"LightSlateBlue" => [132, 112, 255, 0xff],
"medium blue" => [0, 0, 205, 0xff],
"MediumBlue" => [0, 0, 205, 0xff],
"royal blue" => [65, 105, 225, 0xff],
"RoyalBlue" => [65, 105, 225, 0xff],
"blue" => [0, 0, 255, 0xff],
"dodger blue" => [30, 144, 255, 0xff],
"DodgerBlue" => [30, 144, 255, 0xff],
"deep sky blue" => [0, 191, 255, 0xff],
"DeepSkyBlue" => [0, 191, 255, 0xff],
"sky blue" => [135, 206, 235, 0xff],
"SkyBlue" => [135, 206, 235, 0xff],
"light sky blue" => [135, 206, 250, 0xff],
"LightSkyBlue" => [135, 206, 250, 0xff],
"steel blue" => [70, 130, 180, 0xff],
"SteelBlue" => [70, 130, 180, 0xff],
"light steel blue" => [176, 196, 222, 0xff],
"LightSteelBlue" => [176, 196, 222, 0xff],
"light blue" => [173, 216, 230, 0xff],
"LightBlue" => [173, 216, 230, 0xff],
"powder blue" => [176, 224, 230, 0xff],
"PowderBlue" => [176, 224, 230, 0xff],
"pale turquoise" => [175, 238, 238, 0xff],
"PaleTurquoise" => [175, 238, 238, 0xff],
"dark turquoise" => [0, 206, 209, 0xff],
"DarkTurquoise" => [0, 206, 209, 0xff],
"medium turquoise" => [72, 209, 204, 0xff],
"MediumTurquoise" => [72, 209, 204, 0xff],
"turquoise" => [64, 224, 208, 0xff],
"cyan" => [0, 255, 255, 0xff],
"aqua" => [0, 255, 255, 0xff],
"light cyan" => [224, 255, 255, 0xff],
"LightCyan" => [224, 255, 255, 0xff],
"cadet blue" => [95, 158, 160, 0xff],
"CadetBlue" => [95, 158, 160, 0xff],
"medium aquamarine" => [102, 205, 170, 0xff],
"MediumAquamarine" => [102, 205, 170, 0xff],
"aquamarine" => [127, 255, 212, 0xff],
"dark green" => [0, 100, 0, 0xff],
"DarkGreen" => [0, 100, 0, 0xff],
"dark olive green" => [85, 107, 47, 0xff],
"DarkOliveGreen" => [85, 107, 47, 0xff],
"dark sea green" => [143, 188, 143, 0xff],
"DarkSeaGreen" => [143, 188, 143, 0xff],
"sea green" => [46, 139, 87, 0xff],
"SeaGreen" => [46, 139, 87, 0xff],
"medium sea green" => [60, 179, 113, 0xff],
"MediumSeaGreen" => [60, 179, 113, 0xff],
"light sea green" => [32, 178, 170, 0xff],
"LightSeaGreen" => [32, 178, 170, 0xff],
"pale green" => [152, 251, 152, 0xff],
"PaleGreen" => [152, 251, 152, 0xff],
"spring green" => [0, 255, 127, 0xff],
"SpringGreen" => [0, 255, 127, 0xff],
"lawn green" => [124, 252, 0, 0xff],
"LawnGreen" => [124, 252, 0, 0xff],
"green" => [0, 255, 0, 0xff],
"lime" => [0, 255, 0, 0xff],
"x11 green" => [0, 255, 0, 0xff],
"X11Green" => [0, 255, 0, 0xff],
"web green" => [0, 128, 0, 0xff],
"WebGreen" => [0, 128, 0, 0xff],
"chartreuse" => [127, 255, 0, 0xff],
"medium spring green" => [0, 250, 154, 0xff],
"MediumSpringGreen" => [0, 250, 154, 0xff],
"green yellow" => [173, 255, 47, 0xff],
"GreenYellow" => [173, 255, 47, 0xff],
"lime green" => [50, 205, 50, 0xff],
"LimeGreen" => [50, 205, 50, 0xff],
"yellow green" => [154, 205, 50, 0xff],
"YellowGreen" => [154, 205, 50, 0xff],
"forest green" => [34, 139, 34, 0xff],
"ForestGreen" => [34, 139, 34, 0xff],
"olive drab" => [107, 142, 35, 0xff],
"OliveDrab" => [107, 142, 35, 0xff],
"dark khaki" => [189, 183, 107, 0xff],
"DarkKhaki" => [189, 183, 107, 0xff],
"khaki" => [240, 230, 140, 0xff],
"pale goldenrod" => [238, 232, 170, 0xff],
"PaleGoldenrod" => [238, 232, 170, 0xff],
"light goldenrod yellow" => [250, 250, 210, 0xff],
"LightGoldenrodYellow" => [250, 250, 210, 0xff],
"light yellow" => [255, 255, 224, 0xff],
"LightYellow" => [255, 255, 224, 0xff],
"yellow" => [255, 255, 0, 0xff],
"gold" => [255, 215, 0, 0xff],
"light goldenrod" => [238, 221, 130, 0xff],
"LightGoldenrod" => [238, 221, 130, 0xff],
"goldenrod" => [218, 165, 32, 0xff],
"dark goldenrod" => [184, 134, 11, 0xff],
"DarkGoldenrod" => [184, 134, 11, 0xff],
"rosy brown" => [188, 143, 143, 0xff],
"RosyBrown" => [188, 143, 143, 0xff],
"indian red" => [205, 92, 92, 0xff],
"IndianRed" => [205, 92, 92, 0xff],
"saddle brown" => [139, 69, 19, 0xff],
"SaddleBrown" => [139, 69, 19, 0xff],
"sienna" => [160, 82, 45, 0xff],
"peru" => [205, 133, 63, 0xff],
"burlywood" => [222, 184, 135, 0xff],
"beige" => [245, 245, 220, 0xff],
"wheat" => [245, 222, 179, 0xff],
"sandy brown" => [244, 164, 96, 0xff],
"SandyBrown" => [244, 164, 96, 0xff],
"tan" => [210, 180, 140, 0xff],
"chocolate" => [210, 105, 30, 0xff],
"firebrick" => [178, 34, 34, 0xff],
"brown" => [165, 42, 42, 0xff],
"dark salmon" => [233, 150, 122, 0xff],
"DarkSalmon" => [233, 150, 122, 0xff],
"salmon" => [250, 128, 114, 0xff],
"light salmon" => [255, 160, 122, 0xff],
"LightSalmon" => [255, 160, 122, 0xff],
"orange" => [255, 165, 0, 0xff],
"dark orange" => [255, 140, 0, 0xff],
"DarkOrange" => [255, 140, 0, 0xff],
"coral" => [255, 127, 80, 0xff],
"light coral" => [240, 128, 128, 0xff],
"LightCoral" => [240, 128, 128, 0xff],
"tomato" => [255, 99, 71, 0xff],
"orange red" => [255, 69, 0, 0xff],
"OrangeRed" => [255, 69, 0, 0xff],
"red" => [255, 0, 0, 0xff],
"hot pink" => [255, 105, 180, 0xff],
"HotPink" => [255, 105, 180, 0xff],
"deep pink" => [255, 20, 147, 0xff],
"DeepPink" => [255, 20, 147, 0xff],
"pink" => [255, 192, 203, 0xff],
"light pink" => [255, 182, 193, 0xff],
"LightPink" => [255, 182, 193, 0xff],
"pale violet red" => [219, 112, 147, 0xff],
"PaleVioletRed" => [219, 112, 147, 0xff],
"maroon" => [176, 48, 96, 0xff],
"x11 maroon" => [176, 48, 96, 0xff],
"X11Maroon" => [176, 48, 96, 0xff],
"web maroon" => [128, 0, 0, 0xff],
"WebMaroon" => [128, 0, 0, 0xff],
"medium violet red" => [199, 21, 133, 0xff],
"MediumVioletRed" => [199, 21, 133, 0xff],
"violet red" => [208, 32, 144, 0xff],
"VioletRed" => [208, 32, 144, 0xff],
"magenta" => [255, 0, 255, 0xff],
"fuchsia" => [255, 0, 255, 0xff],
"violet" => [238, 130, 238, 0xff],
"plum" => [221, 160, 221, 0xff],
"orchid" => [218, 112, 214, 0xff],
"medium orchid" => [186, 85, 211, 0xff],
"MediumOrchid" => [186, 85, 211, 0xff],
"dark orchid" => [153, 50, 204, 0xff],
"DarkOrchid" => [153, 50, 204, 0xff],
"dark violet" => [148, 0, 211, 0xff],
"DarkViolet" => [148, 0, 211, 0xff],
"blue violet" => [138, 43, 226, 0xff],
"BlueViolet" => [138, 43, 226, 0xff],
"purple" => [160, 32, 240, 0xff],
"x11 purple" => [160, 32, 240, 0xff],
"X11Purple" => [160, 32, 240, 0xff],
"web purple" => [128, 0, 128, 0xff],
"WebPurple" => [128, 0, 128, 0xff],
"medium purple" => [147, 112, 219, 0xff],
"MediumPurple" => [147, 112, 219, 0xff],
"thistle" => [216, 191, 216, 0xff],
"snow1" => [255, 250, 250, 0xff],
"snow2" => [238, 233, 233, 0xff],
"snow3" => [205, 201, 201, 0xff],
"snow4" => [139, 137, 137, 0xff],
"seashell1" => [255, 245, 238, 0xff],
"seashell2" => [238, 229, 222, 0xff],
"seashell3" => [205, 197, 191, 0xff],
"seashell4" => [139, 134, 130, 0xff],
"AntiqueWhite1" => [255, 239, 219, 0xff],
"AntiqueWhite2" => [238, 223, 204, 0xff],
"AntiqueWhite3" => [205, 192, 176, 0xff],
"AntiqueWhite4" => [139, 131, 120, 0xff],
"bisque1" => [255, 228, 196, 0xff],
"bisque2" => [238, 213, 183, 0xff],
"bisque3" => [205, 183, 158, 0xff],
"bisque4" => [139, 125, 107, 0xff],
"PeachPuff1" => [255, 218, 185, 0xff],
"PeachPuff2" => [238, 203, 173, 0xff],
"PeachPuff3" => [205, 175, 149, 0xff],
"PeachPuff4" => [139, 119, 101, 0xff],
"NavajoWhite1" => [255, 222, 173, 0xff],
"NavajoWhite2" => [238, 207, 161, 0xff],
"NavajoWhite3" => [205, 179, 139, 0xff],
"NavajoWhite4" => [139, 121, 94, 0xff],
"LemonChiffon1" => [255, 250, 205, 0xff],
"LemonChiffon2" => [238, 233, 191, 0xff],
"LemonChiffon3" => [205, 201, 165, 0xff],
"LemonChiffon4" => [139, 137, 112, 0xff],
"cornsilk1" => [255, 248, 220, 0xff],
"cornsilk2" => [238, 232, 205, 0xff],
"cornsilk3" => [205, 200, 177, 0xff],
"cornsilk4" => [139, 136, 120, 0xff],
"ivory1" => [255, 255, 240, 0xff],
"ivory2" => [238, 238, 224, 0xff],
"ivory3" => [205, 205, 193, 0xff],
"ivory4" => [139, 139, 131, 0xff],
"honeydew1" => [240, 255, 240, 0xff],
"honeydew2" => [224, 238, 224, 0xff],
"honeydew3" => [193, 205, 193, 0xff],
"honeydew4" => [131, 139, 131, 0xff],
"LavenderBlush1" => [255, 240, 245, 0xff],
"LavenderBlush2" => [238, 224, 229, 0xff],
"LavenderBlush3" => [205, 193, 197, 0xff],
"LavenderBlush4" => [139, 131, 134, 0xff],
"MistyRose1" => [255, 228, 225, 0xff],
"MistyRose2" => [238, 213, 210, 0xff],
"MistyRose3" => [205, 183, 181, 0xff],
"MistyRose4" => [139, 125, 123, 0xff],
"azure1" => [240, 255, 255, 0xff],
"azure2" => [224, 238, 238, 0xff],
"azure3" => [193, 205, 205, 0xff],
"azure4" => [131, 139, 139, 0xff],
"SlateBlue1" => [131, 111, 255, 0xff],
"SlateBlue2" => [122, 103, 238, 0xff],
"SlateBlue3" => [105, 89, 205, 0xff],
"SlateBlue4" => [71, 60, 139, 0xff],
"RoyalBlue1" => [72, 118, 255, 0xff],
"RoyalBlue2" => [67, 110, 238, 0xff],
"RoyalBlue3" => [58, 95, 205, 0xff],
"RoyalBlue4" => [39, 64, 139, 0xff],
"blue1" => [0, 0, 255, 0xff],
"blue2" => [0, 0, 238, 0xff],
"blue3" => [0, 0, 205, 0xff],
"blue4" => [0, 0, 139, 0xff],
"DodgerBlue1" => [30, 144, 255, 0xff],
"DodgerBlue2" => [28, 134, 238, 0xff],
"DodgerBlue3" => [24, 116, 205, 0xff],
"DodgerBlue4" => [16, 78, 139, 0xff],
"SteelBlue1" => [99, 184, 255, 0xff],
"SteelBlue2" => [92, 172, 238, 0xff],
"SteelBlue3" => [79, 148, 205, 0xff],
"SteelBlue4" => [54, 100, 139, 0xff],
"DeepSkyBlue1" => [0, 191, 255, 0xff],
"DeepSkyBlue2" => [0, 178, 238, 0xff],
"DeepSkyBlue3" => [0, 154, 205, 0xff],
"DeepSkyBlue4" => [0, 104, 139, 0xff],
"SkyBlue1" => [135, 206, 255, 0xff],
"SkyBlue2" => [126, 192, 238, 0xff],
"SkyBlue3" => [108, 166, 205, 0xff],
"SkyBlue4" => [74, 112, 139, 0xff],
"LightSkyBlue1" => [176, 226, 255, 0xff],
"LightSkyBlue2" => [164, 211, 238, 0xff],
"LightSkyBlue3" => [141, 182, 205, 0xff],
"LightSkyBlue4" => [96, 123, 139, 0xff],
"SlateGray1" => [198, 226, 255, 0xff],
"SlateGray2" => [185, 211, 238, 0xff],
"SlateGray3" => [159, 182, 205, 0xff],
"SlateGray4" => [108, 123, 139, 0xff],
"LightSteelBlue1" => [202, 225, 255, 0xff],
"LightSteelBlue2" => [188, 210, 238, 0xff],
"LightSteelBlue3" => [162, 181, 205, 0xff],
"LightSteelBlue4" => [110, 123, 139, 0xff],
"LightBlue1" => [191, 239, 255, 0xff],
"LightBlue2" => [178, 223, 238, 0xff],
"LightBlue3" => [154, 192, 205, 0xff],
"LightBlue4" => [104, 131, 139, 0xff],
"LightCyan1" => [224, 255, 255, 0xff],
"LightCyan2" => [209, 238, 238, 0xff],
"LightCyan3" => [180, 205, 205, 0xff],
"LightCyan4" => [122, 139, 139, 0xff],
"PaleTurquoise1" => [187, 255, 255, 0xff],
"PaleTurquoise2" => [174, 238, 238, 0xff],
"PaleTurquoise3" => [150, 205, 205, 0xff],
"PaleTurquoise4" => [102, 139, 139, 0xff],
"CadetBlue1" => [152, 245, 255, 0xff],
"CadetBlue2" => [142, 229, 238, 0xff],
"CadetBlue3" => [122, 197, 205, 0xff],
"CadetBlue4" => [83, 134, 139, 0xff],
"turquoise1" => [0, 245, 255, 0xff],
"turquoise2" => [0, 229, 238, 0xff],
"turquoise3" => [0, 197, 205, 0xff],
"turquoise4" => [0, 134, 139, 0xff],
"cyan1" => [0, 255, 255, 0xff],
"cyan2" => [0, 238, 238, 0xff],
"cyan3" => [0, 205, 205, 0xff],
"cyan4" => [0, 139, 139, 0xff],
"DarkSlateGray1" => [151, 255, 255, 0xff],
"DarkSlateGray2" => [141, 238, 238, 0xff],
"DarkSlateGray3" => [121, 205, 205, 0xff],
"DarkSlateGray4" => [82, 139, 139, 0xff],
"aquamarine1" => [127, 255, 212, 0xff],
"aquamarine2" => [118, 238, 198, 0xff],
"aquamarine3" => [102, 205, 170, 0xff],
"aquamarine4" => [69, 139, 116, 0xff],
"DarkSeaGreen1" => [193, 255, 193, 0xff],
"DarkSeaGreen2" => [180, 238, 180, 0xff],
"DarkSeaGreen3" => [155, 205, 155, 0xff],
"DarkSeaGreen4" => [105, 139, 105, 0xff],
"SeaGreen1" => [84, 255, 159, 0xff],
"SeaGreen2" => [78, 238, 148, 0xff],
"SeaGreen3" => [67, 205, 128, 0xff],
"SeaGreen4" => [46, 139, 87, 0xff],
"PaleGreen1" => [154, 255, 154, 0xff],
"PaleGreen2" => [144, 238, 144, 0xff],
"PaleGreen3" => [124, 205, 124, 0xff],
"PaleGreen4" => [84, 139, 84, 0xff],
"SpringGreen1" => [0, 255, 127, 0xff],
"SpringGreen2" => [0, 238, 118, 0xff],
"SpringGreen3" => [0, 205, 102, 0xff],
"SpringGreen4" => [0, 139, 69, 0xff],
"green1" => [0, 255, 0, 0xff],
"green2" => [0, 238, 0, 0xff],
"green3" => [0, 205, 0, 0xff],
"green4" => [0, 139, 0, 0xff],
"chartreuse1" => [127, 255, 0, 0xff],
"chartreuse2" => [118, 238, 0, 0xff],
"chartreuse3" => [102, 205, 0, 0xff],
"chartreuse4" => [69, 139, 0, 0xff],
"OliveDrab1" => [192, 255, 62, 0xff],
"OliveDrab2" => [179, 238, 58, 0xff],
"OliveDrab3" => [154, 205, 50, 0xff],
"OliveDrab4" => [105, 139, 34, 0xff],
"DarkOliveGreen1" => [202, 255, 112, 0xff],
"DarkOliveGreen2" => [188, 238, 104, 0xff],
"DarkOliveGreen3" => [162, 205, 90, 0xff],
"DarkOliveGreen4" => [110, 139, 61, 0xff],
"khaki1" => [255, 246, 143, 0xff],
"khaki2" => [238, 230, 133, 0xff],
"khaki3" => [205, 198, 115, 0xff],
"khaki4" => [139, 134, 78, 0xff],
"LightGoldenrod1" => [255, 236, 139, 0xff],
"LightGoldenrod2" => [238, 220, 130, 0xff],
"LightGoldenrod3" => [205, 190, 112, 0xff],
"LightGoldenrod4" => [139, 129, 76, 0xff],
"LightYellow1" => [255, 255, 224, 0xff],
"LightYellow2" => [238, 238, 209, 0xff],
"LightYellow3" => [205, 205, 180, 0xff],
"LightYellow4" => [139, 139, 122, 0xff],
"yellow1" => [255, 255, 0, 0xff],
"yellow2" => [238, 238, 0, 0xff],
"yellow3" => [205, 205, 0, 0xff],
"yellow4" => [139, 139, 0, 0xff],
"gold1" => [255, 215, 0, 0xff],
"gold2" => [238, 201, 0, 0xff],
"gold3" => [205, 173, 0, 0xff],
"gold4" => [139, 117, 0, 0xff],
"goldenrod1" => [255, 193, 37, 0xff],
"goldenrod2" => [238, 180, 34, 0xff],
"goldenrod3" => [205, 155, 29, 0xff],
"goldenrod4" => [139, 105, 20, 0xff],
"DarkGoldenrod1" => [255, 185, 15, 0xff],
"DarkGoldenrod2" => [238, 173, 14, 0xff],
"DarkGoldenrod3" => [205, 149, 12, 0xff],
"DarkGoldenrod4" => [139, 101, 8, 0xff],
"RosyBrown1" => [255, 193, 193, 0xff],
"RosyBrown2" => [238, 180, 180, 0xff],
"RosyBrown3" => [205, 155, 155, 0xff],
"RosyBrown4" => [139, 105, 105, 0xff],
"IndianRed1" => [255, 106, 106, 0xff],
"IndianRed2" => [238, 99, 99, 0xff],
"IndianRed3" => [205, 85, 85, 0xff],
"IndianRed4" => [139, 58, 58, 0xff],
"sienna1" => [255, 130, 71, 0xff],
"sienna2" => [238, 121, 66, 0xff],
"sienna3" => [205, 104, 57, 0xff],
"sienna4" => [139, 71, 38, 0xff],
"burlywood1" => [255, 211, 155, 0xff],
"burlywood2" => [238, 197, 145, 0xff],
"burlywood3" => [205, 170, 125, 0xff],
"burlywood4" => [139, 115, 85, 0xff],
"wheat1" => [255, 231, 186, 0xff],
"wheat2" => [238, 216, 174, 0xff],
"wheat3" => [205, 186, 150, 0xff],
"wheat4" => [139, 126, 102, 0xff],
"tan1" => [255, 165, 79, 0xff],
"tan2" => [238, 154, 73, 0xff],
"tan3" => [205, 133, 63, 0xff],
"tan4" => [139, 90, 43, 0xff],
"chocolate1" => [255, 127, 36, 0xff],
"chocolate2" => [238, 118, 33, 0xff],
"chocolate3" => [205, 102, 29, 0xff],
"chocolate4" => [139, 69, 19, 0xff],
"firebrick1" => [255, 48, 48, 0xff],
"firebrick2" => [238, 44, 44, 0xff],
"firebrick3" => [205, 38, 38, 0xff],
"firebrick4" => [139, 26, 26, 0xff],
"brown1" => [255, 64, 64, 0xff],
"brown2" => [238, 59, 59, 0xff],
"brown3" => [205, 51, 51, 0xff],
"brown4" => [139, 35, 35, 0xff],
"salmon1" => [255, 140, 105, 0xff],
"salmon2" => [238, 130, 98, 0xff],
"salmon3" => [205, 112, 84, 0xff],
"salmon4" => [139, 76, 57, 0xff],
"LightSalmon1" => [255, 160, 122, 0xff],
"LightSalmon2" => [238, 149, 114, 0xff],
"LightSalmon3" => [205, 129, 98, 0xff],
"LightSalmon4" => [139, 87, 66, 0xff],
"orange1" => [255, 165, 0, 0xff],
"orange2" => [238, 154, 0, 0xff],
"orange3" => [205, 133, 0, 0xff],
"orange4" => [139, 90, 0, 0xff],
"DarkOrange1" => [255, 127, 0, 0xff],
"DarkOrange2" => [238, 118, 0, 0xff],
"DarkOrange3" => [205, 102, 0, 0xff],
"DarkOrange4" => [139, 69, 0, 0xff],
"coral1" => [255, 114, 86, 0xff],
"coral2" => [238, 106, 80, 0xff],
"coral3" => [205, 91, 69, 0xff],
"coral4" => [139, 62, 47, 0xff],
"tomato1" => [255, 99, 71, 0xff],
"tomato2" => [238, 92, 66, 0xff],
"tomato3" => [205, 79, 57, 0xff],
"tomato4" => [139, 54, 38, 0xff],
"OrangeRed1" => [255, 69, 0, 0xff],
"OrangeRed2" => [238, 64, 0, 0xff],
"OrangeRed3" => [205, 55, 0, 0xff],
"OrangeRed4" => [139, 37, 0, 0xff],
"red1" => [255, 0, 0, 0xff],
"red2" => [238, 0, 0, 0xff],
"red3" => [205, 0, 0, 0xff],
"red4" => [139, 0, 0, 0xff],
"DeepPink1" => [255, 20, 147, 0xff],
"DeepPink2" => [238, 18, 137, 0xff],
"DeepPink3" => [205, 16, 118, 0xff],
"DeepPink4" => [139, 10, 80, 0xff],
"HotPink1" => [255, 110, 180, 0xff],
"HotPink2" => [238, 106, 167, 0xff],
"HotPink3" => [205, 96, 144, 0xff],
"HotPink4" => [139, 58, 98, 0xff],
"pink1" => [255, 181, 197, 0xff],
"pink2" => [238, 169, 184, 0xff],
"pink3" => [205, 145, 158, 0xff],
"pink4" => [139, 99, 108, 0xff],
"LightPink1" => [255, 174, 185, 0xff],
"LightPink2" => [238, 162, 173, 0xff],
"LightPink3" => [205, 140, 149, 0xff],
"LightPink4" => [139, 95, 101, 0xff],
"PaleVioletRed1" => [255, 130, 171, 0xff],
"PaleVioletRed2" => [238, 121, 159, 0xff],
"PaleVioletRed3" => [205, 104, 137, 0xff],
"PaleVioletRed4" => [139, 71, 93, 0xff],
"maroon1" => [255, 52, 179, 0xff],
"maroon2" => [238, 48, 167, 0xff],
"maroon3" => [205, 41, 144, 0xff],
"maroon4" => [139, 28, 98, 0xff],
"VioletRed1" => [255, 62, 150, 0xff],
"VioletRed2" => [238, 58, 140, 0xff],
"VioletRed3" => [205, 50, 120, 0xff],
"VioletRed4" => [139, 34, 82, 0xff],
"magenta1" => [255, 0, 255, 0xff],
"magenta2" => [238, 0, 238, 0xff],
"magenta3" => [205, 0, 205, 0xff],
"magenta4" => [139, 0, 139, 0xff],
"orchid1" => [255, 131, 250, 0xff],
"orchid2" => [238, 122, 233, 0xff],
"orchid3" => [205, 105, 201, 0xff],
"orchid4" => [139, 71, 137, 0xff],
"plum1" => [255, 187, 255, 0xff],
"plum2" => [238, 174, 238, 0xff],
"plum3" => [205, 150, 205, 0xff],
"plum4" => [139, 102, 139, 0xff],
"MediumOrchid1" => [224, 102, 255, 0xff],
"MediumOrchid2" => [209, 95, 238, 0xff],
"MediumOrchid3" => [180, 82, 205, 0xff],
"MediumOrchid4" => [122, 55, 139, 0xff],
"DarkOrchid1" => [191, 62, 255, 0xff],
"DarkOrchid2" => [178, 58, 238, 0xff],
"DarkOrchid3" => [154, 50, 205, 0xff],
"DarkOrchid4" => [104, 34, 139, 0xff],
"purple1" => [155, 48, 255, 0xff],
"purple2" => [145, 44, 238, 0xff],
"purple3" => [125, 38, 205, 0xff],
"purple4" => [85, 26, 139, 0xff],
"MediumPurple1" => [171, 130, 255, 0xff],
"MediumPurple2" => [159, 121, 238, 0xff],
"MediumPurple3" => [137, 104, 205, 0xff],
"MediumPurple4" => [93, 71, 139, 0xff],
"thistle1" => [255, 225, 255, 0xff],
"thistle2" => [238, 210, 238, 0xff],
"thistle3" => [205, 181, 205, 0xff],
"thistle4" => [139, 123, 139, 0xff],
"gray0" => [0, 0, 0, 0xff],
"grey0" => [0, 0, 0, 0xff],
"gray1" => [3, 3, 3, 0xff],
"grey1" => [3, 3, 3, 0xff],
"gray2" => [5, 5, 5, 0xff],
"grey2" => [5, 5, 5, 0xff],
"gray3" => [8, 8, 8, 0xff],
"grey3" => [8, 8, 8, 0xff],
"gray4" => [10, 10, 10, 0xff],
"grey4" => [10, 10, 10, 0xff],
"gray5" => [13, 13, 13, 0xff],
"grey5" => [13, 13, 13, 0xff],
"gray6" => [15, 15, 15, 0xff],
"grey6" => [15, 15, 15, 0xff],
"gray7" => [18, 18, 18, 0xff],
"grey7" => [18, 18, 18, 0xff],
"gray8" => [20, 20, 20, 0xff],
"grey8" => [20, 20, 20, 0xff],
"gray9" => [23, 23, 23, 0xff],
"grey9" => [23, 23, 23, 0xff],
"gray10" => [26, 26, 26, 0xff],
"grey10" => [26, 26, 26, 0xff],
"gray11" => [28, 28, 28, 0xff],
"grey11" => [28, 28, 28, 0xff],
"gray12" => [31, 31, 31, 0xff],
"grey12" => [31, 31, 31, 0xff],
"gray13" => [33, 33, 33, 0xff],
"grey13" => [33, 33, 33, 0xff],
"gray14" => [36, 36, 36, 0xff],
"grey14" => [36, 36, 36, 0xff],
"gray15" => [38, 38, 38, 0xff],
"grey15" => [38, 38, 38, 0xff],
"gray16" => [41, 41, 41, 0xff],
"grey16" => [41, 41, 41, 0xff],
"gray17" => [43, 43, 43, 0xff],
"grey17" => [43, 43, 43, 0xff],
"gray18" => [46, 46, 46, 0xff],
"grey18" => [46, 46, 46, 0xff],
"gray19" => [48, 48, 48, 0xff],
"grey19" => [48, 48, 48, 0xff],
"gray20" => [51, 51, 51, 0xff],
"grey20" => [51, 51, 51, 0xff],
"gray21" => [54, 54, 54, 0xff],
"grey21" => [54, 54, 54, 0xff],
"gray22" => [56, 56, 56, 0xff],
"grey22" => [56, 56, 56, 0xff],
"gray23" => [59, 59, 59, 0xff],
"grey23" => [59, 59, 59, 0xff],
"gray24" => [61, 61, 61, 0xff],
"grey24" => [61, 61, 61, 0xff],
"gray25" => [64, 64, 64, 0xff],
"grey25" => [64, 64, 64, 0xff],
"gray26" => [66, 66, 66, 0xff],
"grey26" => [66, 66, 66, 0xff],
"gray27" => [69, 69, 69, 0xff],
"grey27" => [69, 69, 69, 0xff],
"gray28" => [71, 71, 71, 0xff],
"grey28" => [71, 71, 71, 0xff],
"gray29" => [74, 74, 74, 0xff],
"grey29" => [74, 74, 74, 0xff],
"gray30" => [77, 77, 77, 0xff],
"grey30" => [77, 77, 77, 0xff],
"gray31" => [79, 79, 79, 0xff],
"grey31" => [79, 79, 79, 0xff],
"gray32" => [82, 82, 82, 0xff],
"grey32" => [82, 82, 82, 0xff],
"gray33" => [84, 84, 84, 0xff],
"grey33" => [84, 84, 84, 0xff],
"gray34" => [87, 87, 87, 0xff],
"grey34" => [87, 87, 87, 0xff],
"gray35" => [89, 89, 89, 0xff],
"grey35" => [89, 89, 89, 0xff],
"gray36" => [92, 92, 92, 0xff],
"grey36" => [92, 92, 92, 0xff],
"gray37" => [94, 94, 94, 0xff],
"grey37" => [94, 94, 94, 0xff],
"gray38" => [97, 97, 97, 0xff],
"grey38" => [97, 97, 97, 0xff],
"gray39" => [99, 99, 99, 0xff],
"grey39" => [99, 99, 99, 0xff],
"gray40" => [102, 102, 102, 0xff],
"grey40" => [102, 102, 102, 0xff],
"gray41" => [105, 105, 105, 0xff],
"grey41" => [105, 105, 105, 0xff],
"gray42" => [107, 107, 107, 0xff],
"grey42" => [107, 107, 107, 0xff],
"gray43" => [110, 110, 110, 0xff],
"grey43" => [110, 110, 110, 0xff],
"gray44" => [112, 112, 112, 0xff],
"grey44" => [112, 112, 112, 0xff],
"gray45" => [115, 115, 115, 0xff],
"grey45" => [115, 115, 115, 0xff],
"gray46" => [117, 117, 117, 0xff],
"grey46" => [117, 117, 117, 0xff],
"gray47" => [120, 120, 120, 0xff],
"grey47" => [120, 120, 120, 0xff],
"gray48" => [122, 122, 122, 0xff],
"grey48" => [122, 122, 122, 0xff],
"gray49" => [125, 125, 125, 0xff],
"grey49" => [125, 125, 125, 0xff],
"gray50" => [127, 127, 127, 0xff],
"grey50" => [127, 127, 127, 0xff],
"gray51" => [130, 130, 130, 0xff],
"grey51" => [130, 130, 130, 0xff],
"gray52" => [133, 133, 133, 0xff],
"grey52" => [133, 133, 133, 0xff],
"gray53" => [135, 135, 135, 0xff],
"grey53" => [135, 135, 135, 0xff],
"gray54" => [138, 138, 138, 0xff],
"grey54" => [138, 138, 138, 0xff],
"gray55" => [140, 140, 140, 0xff],
"grey55" => [140, 140, 140, 0xff],
"gray56" => [143, 143, 143, 0xff],
"grey56" => [143, 143, 143, 0xff],
"gray57" => [145, 145, 145, 0xff],
"grey57" => [145, 145, 145, 0xff],
"gray58" => [148, 148, 148, 0xff],
"grey58" => [148, 148, 148, 0xff],
"gray59" => [150, 150, 150, 0xff],
"grey59" => [150, 150, 150, 0xff],
"gray60" => [153, 153, 153, 0xff],
"grey60" => [153, 153, 153, 0xff],
"gray61" => [156, 156, 156, 0xff],
"grey61" => [156, 156, 156, 0xff],
"gray62" => [158, 158, 158, 0xff],
"grey62" => [158, 158, 158, 0xff],
"gray63" => [161, 161, 161, 0xff],
"grey63" => [161, 161, 161, 0xff],
"gray64" => [163, 163, 163, 0xff],
"grey64" => [163, 163, 163, 0xff],
"gray65" => [166, 166, 166, 0xff],
"grey65" => [166, 166, 166, 0xff],
"gray66" => [168, 168, 168, 0xff],
"grey66" => [168, 168, 168, 0xff],
"gray67" => [171, 171, 171, 0xff],
"grey67" => [171, 171, 171, 0xff],
"gray68" => [173, 173, 173, 0xff],
"grey68" => [173, 173, 173, 0xff],
"gray69" => [176, 176, 176, 0xff],
"grey69" => [176, 176, 176, 0xff],
"gray70" => [179, 179, 179, 0xff],
"grey70" => [179, 179, 179, 0xff],
"gray71" => [181, 181, 181, 0xff],
"grey71" => [181, 181, 181, 0xff],
"gray72" => [184, 184, 184, 0xff],
"grey72" => [184, 184, 184, 0xff],
"gray73" => [186, 186, 186, 0xff],
"grey73" => [186, 186, 186, 0xff],
"gray74" => [189, 189, 189, 0xff],
"grey74" => [189, 189, 189, 0xff],
"gray75" => [191, 191, 191, 0xff],
"grey75" => [191, 191, 191, 0xff],
"gray76" => [194, 194, 194, 0xff],
"grey76" => [194, 194, 194, 0xff],
"gray77" => [196, 196, 196, 0xff],
"grey77" => [196, 196, 196, 0xff],
"gray78" => [199, 199, 199, 0xff],
"grey78" => [199, 199, 199, 0xff],
"gray79" => [201, 201, 201, 0xff],
"grey79" => [201, 201, 201, 0xff],
"gray80" => [204, 204, 204, 0xff],
"grey80" => [204, 204, 204, 0xff],
"gray81" => [207, 207, 207, 0xff],
"grey81" => [207, 207, 207, 0xff],
"gray82" => [209, 209, 209, 0xff],
"grey82" => [209, 209, 209, 0xff],
"gray83" => [212, 212, 212, 0xff],
"grey83" => [212, 212, 212, 0xff],
"gray84" => [214, 214, 214, 0xff],
"grey84" => [214, 214, 214, 0xff],
"gray85" => [217, 217, 217, 0xff],
"grey85" => [217, 217, 217, 0xff],
"gray86" => [219, 219, 219, 0xff],
"grey86" => [219, 219, 219, 0xff],
"gray87" => [222, 222, 222, 0xff],
"grey87" => [222, 222, 222, 0xff],
"gray88" => [224, 224, 224, 0xff],
"grey88" => [224, 224, 224, 0xff],
"gray89" => [227, 227, 227, 0xff],
"grey89" => [227, 227, 227, 0xff],
"gray90" => [229, 229, 229, 0xff],
"grey90" => [229, 229, 229, 0xff],
"gray91" => [232, 232, 232, 0xff],
"grey91" => [232, 232, 232, 0xff],
"gray92" => [235, 235, 235, 0xff],
"grey92" => [235, 235, 235, 0xff],
"gray93" => [237, 237, 237, 0xff],
"grey93" => [237, 237, 237, 0xff],
"gray94" => [240, 240, 240, 0xff],
"grey94" => [240, 240, 240, 0xff],
"gray95" => [242, 242, 242, 0xff],
"grey95" => [242, 242, 242, 0xff],
"gray96" => [245, 245, 245, 0xff],
"grey96" => [245, 245, 245, 0xff],
"gray97" => [247, 247, 247, 0xff],
"grey97" => [247, 247, 247, 0xff],
"gray98" => [250, 250, 250, 0xff],
"grey98" => [250, 250, 250, 0xff],
"gray99" => [252, 252, 252, 0xff],
"grey99" => [252, 252, 252, 0xff],
"gray100" => [255, 255, 255, 0xff],
"grey100" => [255, 255, 255, 0xff],
"dark grey" => [169, 169, 169, 0xff],
"DarkGrey" => [169, 169, 169, 0xff],
"dark gray" => [169, 169, 169, 0xff],
"DarkGray" => [169, 169, 169, 0xff],
"dark blue" => [0, 0, 139, 0xff],
"DarkBlue" => [0, 0, 139, 0xff],
"dark cyan" => [0, 139, 139, 0xff],
"DarkCyan" => [0, 139, 139, 0xff],
"dark magenta" => [139, 0, 139, 0xff],
"DarkMagenta" => [139, 0, 139, 0xff],
"dark red" => [139, 0, 0, 0xff],
"DarkRed" => [139, 0, 0, 0xff],
"light green" => [144, 238, 144, 0xff],
"LightGreen" => [144, 238, 144, 0xff],
"crimson" => [220, 20, 60, 0xff],
"indigo" => [75, 0, 130, 0xff],
"olive" => [128, 128, 0, 0xff],
"rebecca purple" => [102, 51, 153, 0xff],
"RebeccaPurple" => [102, 51, 153, 0xff],
"silver" => [192, 192, 192, 0xff],
"teal" => [0, 128, 128, 0xff]
};