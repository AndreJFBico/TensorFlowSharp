
// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System;
using TensorFlow;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using CoreGraphics;
using UIKit;

// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System;
using TensorFlow;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using CoreGraphics;
using UIKit;
using CoreImage;
using ImageIO;
using Foundation;
using ObjCRuntime;

namespace FastStyleTransfer
{
	public static class IOS_IMAGE_LOAD_EXTENSIONS
	{
		public static CGImage ConvertToNative (byte [] buffer, int width, int height)
		{
			using (var pool = new NSAutoreleasePool ()) {
				CGImage img = null;

				unsafe
				{
					fixed (byte* data = &buffer [0]) {
						using (var srcContext = new CGBitmapContext ((IntPtr)data, width, height, 8, width * 4, CGColorSpace.CreateDeviceRGB (), CGImageAlphaInfo.PremultipliedLast)) {
							img = srcContext.ToImage ();
						}
					}
				}

				return img;
			}
		}

		public static byte [] ConvertFromNative (CGImage src)
		{
			var image_width = (int)src.Width;
			var image_height = (int)src.Height;

			try {
				using (var pool = new Foundation.NSAutoreleasePool ()) {
					var buffer = new byte [image_width * image_height * 4];

					unsafe
					{
						fixed (byte* data = &buffer [0]) {
							using (var colorSpace = CGColorSpace.CreateDeviceRGB ()) {
								using (var context = new CGBitmapContext ((IntPtr)data, image_width, image_height, 8, image_width * 4, colorSpace, CGImageAlphaInfo.PremultipliedLast)) {
									// draw the image to the bitmap context
									// this fills the data variable with the image data
									context.InterpolationQuality = CGInterpolationQuality.High;
									context.DrawImage (new CGRect (0, 0, image_width, image_height), src);
								}
							}
						}
					}

					return buffer;
				}
			} catch (Exception e) {
				throw new Exception ("Convert from Native Exception", e);
			}
		}

		public static byte [] ConvertFromNative (CGImage src, int newWidth, int newHeight)
		{
			try {
				using (var pool = new Foundation.NSAutoreleasePool ()) {
					var buffer = new byte [newWidth * newHeight * 4];

					unsafe
					{
						fixed (byte* data = &buffer [0]) {
							using (var colorSpace = CGColorSpace.CreateDeviceRGB ()) {
								using (var context = new CGBitmapContext ((IntPtr)data, newWidth, newHeight, 8, newWidth * 4, colorSpace, CGImageAlphaInfo.PremultipliedLast)) {
									// draw the image to the bitmap context
									// this fills the data variable with the image data
									context.InterpolationQuality = CGInterpolationQuality.High;
									context.DrawImage (new CGRect (0, 0, newWidth, newHeight), src);
								}
							}
						}
					}

					return buffer;
				}
			} catch (Exception e) {
				throw new Exception ("Convert from Native Exception", e);
			}
		}

		public static byte [] LoadImageFromFile (string file_name, out int image_width, out int image_height, out int image_channels)
		{
			var img = UIImage.FromBundle (file_name);

			image_width = (int)img.CGImage.Width;
			image_height = (int)img.CGImage.Height;
			image_channels = 4;
			return ConvertFromNative (img.CGImage, image_width, image_height);
		}
	}
}
