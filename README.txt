//----------------------------------------------------------------------------
// IMAGES
//----------------------------------------------------------------------------

The images were obtained by recording a scene using an ASUS XTION PRO device.
RGB and depth streams are available. Depth is epxressed in milimeters and saved
in unsigned shot images (i.e. 16bit unsigned intergers).

//----------------------------------------------------------------------------
// Calibration data
//----------------------------------------------------------------------------

To do the reconstruction you will need the calibration parameters for the 
cameras.

* RGB camera matrix
    537.9718    0           318.0137
    0           539.6552    267.3932
    0           0           1

* Depth camera matrix
    569.9554    0           325.0815
    0           572.3427    274.5993
    0           0           1

* Pose of the depth camera in the RGB camera coordinate system
  
  Rotation
    [0.9999620314284999, -0.00648372889230588, 0.005822109672519291;
     0.006465051014202089, 0.9999739131377515, 0.003221204643657995;
    -0.005842843209562383, -0.003183442103076048, 0.9999778631947833]

  Translation
    [-0.02695151131463866; -0.0001685908034308943; -0.0003479191573571088]

Note that the lens distortion is negligible for these cameras.