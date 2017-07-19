// An example for using the TensorFlow C# API for style transfer in iOS
// 
// The function Run_TensorFlowStyleTransfer (tf_session, cgImg, 128, 128)
// Requires a power of two width and height.
// If you try it with 1024/1024 it will give you a higher quality image.
//
// The style_quantized model was retrieved from:
// https://github.com/googlecodelabs/tensorflow-style-transfer-android

using System;
using UIKit;
using Foundation;
using AVFoundation;
using CoreFoundation;
using CoreGraphics;
using CoreMedia;
using CoreVideo;
using TensorFlow;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Threading;
using System.Linq;

namespace FastStyleTransfer
{
	public partial class ViewController : UIViewController
	{
		public AVCaptureSession av_session;
		OutputRecorder outputRecorder;
		DispatchQueue queue;
		static UIImageView imgView;

		public ViewController (IntPtr handle) : base (handle)
		{
		}

		#region Aux Structures
		struct RGB
		{
			public byte R;
			public byte G;
			public byte B;
		}

		struct RGBA
		{
			public byte R;
			public byte G;
			public byte B;
			public byte A;
		}
		#endregion

		public async override void ViewDidLoad ()
		{
			base.ViewDidLoad ();

			// Do any additional setup after loading the view.
			imgView = new UIImageView (this.View.Bounds);

			imgView.AutoresizingMask = UIViewAutoresizing.FlexibleBottomMargin | UIViewAutoresizing.FlexibleHeight | UIViewAutoresizing.FlexibleRightMargin | UIViewAutoresizing.FlexibleLeftMargin | UIViewAutoresizing.FlexibleTopMargin | UIViewAutoresizing.FlexibleWidth;
			imgView.ContentMode = UIViewContentMode.ScaleAspectFit;


			View = imgView;

			//Start capture images
			SetupCaptureSession ();

			//Execute Run loop
			await Run ();
		}

		async Task Run ()
		{
			//Load model and setup 
			var tf_session = await Setup_TensorFlowStyleTransfer ();

			while (true) {

				//Get last image of the camera
				var img = await outputRecorder.Pop ();

				if (img == null)
					continue;
				
				var cgImg = img.CGImage;

				if (cgImg != null) {

					var oldImg = imgView.Image;

					//Compute a new image with new style
					//Size must be power of 2
					imgView.Image = await Run_TensorFlowStyleTransfer (tf_session, cgImg, 128, 128);

					using (var pool = new NSAutoreleasePool ()) {
						oldImg?.Dispose ();
						img.Dispose ();
						cgImg.Dispose ();
					}

				} else {
					using (var pool = new NSAutoreleasePool ()) {
						img.Dispose ();
						cgImg.Dispose ();
					}
				}
			}
		}



		static TFTensor CreateTensorFromCameraUImage (CGImage img, int wanted_width, int wanted_height, int wanted_channels, float input_mean, float input_std)
		{
			unsafe
			{
				int image_width = (int)img.Width;
				int image_height = (int)img.Height;
				int image_channels = 4;

				var image_tensor = new TFTensor (TFDataType.Float, new long [] { 1, wanted_width, wanted_height, wanted_channels }, wanted_width * wanted_height * wanted_channels * sizeof (float));

				//Get a pointer to the start
				using (var pool = new NSAutoreleasePool ()) {
					using (var data = img.DataProvider.CopyData ()) {
						byte* in_image_data = (byte*)data.Bytes;
						//Pointer to end
						byte* in_end_image_data = (in_image_data + (image_height * image_width * image_channels));

						//Pointer to tensor data
						float* out_image_data = (float*)image_tensor.Data.ToPointer ();

						for (int y = 0; y < wanted_height; ++y) {
							int in_y = (y * image_height) / wanted_height;
							byte* in_row = in_image_data + (in_y * image_width * image_channels);

							float* out_row = out_image_data + (y * wanted_width * wanted_channels);
							for (int x = 0; x < wanted_width; ++x) {
								int in_x = (x * image_width) / wanted_width;
								byte* in_pixel = in_row + (in_x * image_channels);

								float* out_pixel = out_row + (x * wanted_channels);

								for (int c = 0; c < wanted_channels; ++c) {
									out_pixel [c] = (in_pixel [c] - input_mean) / input_std;
									var j = out_pixel [c];
									var s = (float)out_pixel [c];
								}
							}
						}

					}
				}
				return image_tensor;
			}
		}

		static unsafe byte [] TensorToRGBAarr (int width, int height, TFTensor tensor)
		{
			//var count = width * height * 4;
			var count = (int)tensor.TensorByteSize.ToUInt32 () / sizeof (float);

			byte [] arrayBuffer = new byte [count];

			var pointer = ((float*)tensor.Data.ToPointer ());
			for (int i = 0; i < count; ++i) {
				float value = *pointer * 255f;
				int n;
				if (value < 0) n = 0;
				else if (value > 255) n = 255;
				else n = (int)value;
				arrayBuffer [i] = (byte)n;
				pointer++;
			}

			byte [] rgba = new byte [width * height * 4];
			fixed (byte* rgbaPtr = rgba) {
				fixed (byte* arrayBufferPtr = arrayBuffer) {
					var RGBAptr = (RGB*)rgbaPtr;
					var tmpArrayBufferPtr = (RGB*)arrayBufferPtr;
					for (int i = 0; i < width * height; ++i) {
						*RGBAptr = *tmpArrayBufferPtr;
						((RGBA*)RGBAptr)->A = 255;
						tmpArrayBufferPtr++;

						RGBAptr = (RGB*)IntPtr.Add ((IntPtr)RGBAptr, sizeof (RGBA));
					}
				}
			}

			GC.KeepAlive (tensor);
			return rgba;
		}


		static UIImage TensorToUIImage (TFTensor image_tensor, int image_height, int image_width, int image_channels)
		{
			unsafe
			{
				var arr = TensorToRGBAarr (image_width, image_height, image_tensor);

				CGImage img = null;

				using (var pool = new NSAutoreleasePool ()) {
					unsafe
					{
						fixed (byte* data = &arr [0]) {
							using (var srcContext = new CGBitmapContext ((IntPtr)data, image_width, image_height, 8, image_width * 4, CGColorSpace.CreateDeviceRGB (), CGImageAlphaInfo.PremultipliedLast)) {
								img = srcContext.ToImage ();
							}
						}
					}
				}

				return new UIImage (img, 1, UIImageOrientation.Right);
			}

		}

		private async Task<UIImage> Run_TensorFlowStyleTransfer (TFSession tf_session, CGImage cgImg, int wanted_width, int wanted_height)
		{
			int wanted_channels = 3;
			const float input_mean = 0.0f;
			const float input_std = 255.0f;

			// Run inference on the image files
			// For multiple images, session.Run() can be called in a loop (and
			// concurrently). Alternatively, images can be batched since the model
			// accepts batches of image data as input.

			var stylevals = new float [26];

			for (int f = 0; f < 26; f++) {
				stylevals [f] = 0.0f;
			}
			stylevals [4] = 1.0f;

			UIImage tmpTensorflowImage = null;
			await Task.Run (() => {
				var tensor = CreateTensorFromCameraUImage (cgImg, wanted_width, wanted_height, wanted_channels, input_mean, input_std);

				var styleTensor = new TFTensor (stylevals);
				var graph = tf_session.Graph;

				var runner = tf_session.GetRunner ();
				runner.AddInput (graph ["input"] [0], tensor);
				var g = graph ["style_num"] [0];
				runner.AddInput (g, new TFTensor (stylevals));

				runner.Fetch (graph ["transformer/expand/conv3/conv/Sigmoid"] [0]);

				//########################
				//Bug!!!!! - seems to deadlock once in a while. We can increase its probability by working with smaller images like 16 x 16 
				//########################

				var output = runner.Run ();

				// output[0].Value() is a vector containing probabilities of
				// labels for each image in thestyle_num "batch". The batch size was 1.
				// Find the most probably label index.

				var result = output [0];
				var rshape = result.Shape;
				tmpTensorflowImage = TensorToUIImage (result, wanted_height, wanted_width, wanted_channels);

				foreach(var o in output){
					o?.Dispose ();
				}

				tensor.Dispose ();


			});


			return tmpTensorflowImage;
		}

		private Task<TFSession> Setup_TensorFlowStyleTransfer ()
		{
			
			return Task.Run (() => {
				// Construct an in-memory graph from the serialized form.
				var graph = new TFGraph ();

				var modelFile = Path.Combine ("stylize", "stylize_quantized.pb");

				// Load the serialized GraphDef from a file.
				var modelBuffer = File.ReadAllBytes (modelFile);

				//Import model
				graph.Import (modelBuffer, "");

				return new TFSession (graph);
			});
		}



		public override void ViewWillDisappear (bool animated)
		{
			base.ViewWillDisappear (animated);
			av_session.StopRunning ();
		}

		bool SetupCaptureSession ()
		{
			// configure the capture session for low resolution, change this if your code
			// can cope with more data or volume
			av_session = new AVCaptureSession {
				SessionPreset = AVCaptureSession.PresetMedium
			};

			// create a device input and attach it to the session
			var captureDevice = AVCaptureDevice.GetDefaultDevice (AVMediaType.Video);
			if (captureDevice == null) {
				Console.WriteLine ("No captureDevice - this won't work on the simulator, try a physical device");
				return false;
			}
			//Configure for 15 FPS. Note use of LockForConigfuration()/UnlockForConfiguration()
			NSError error = null;
			captureDevice.LockForConfiguration (out error);
			if (error != null) {
				Console.WriteLine (error);
				captureDevice.UnlockForConfiguration ();
				return false;
			}

			captureDevice.UnlockForConfiguration ();

			var input = AVCaptureDeviceInput.FromDevice (captureDevice);
			if (input == null) {
				Console.WriteLine ("No input - this won't work on the simulator, try a physical device");
				return false;
			}

			av_session.AddInput (input);

			// create a VideoDataOutput and add it to the sesion
			var settings = new CVPixelBufferAttributes {
				PixelFormatType = CVPixelFormatType.CV32BGRA
			};
			using (var output = new AVCaptureVideoDataOutput { WeakVideoSettings = settings.Dictionary }) {
				queue = new DispatchQueue ("myQueue");
				outputRecorder = new OutputRecorder ();
				output.SetSampleBufferDelegate (outputRecorder, queue);
				av_session.AddOutput (output);
			}

			av_session.StartRunning ();

			return true;
		}

		public class OutputRecorder : AVCaptureVideoDataOutputSampleBufferDelegate
		{
			//TaskCompletionSource<UIImage> tcs = new TaskCompletionSource<UIImage> ();

			UIImage currentCamImg;

			static SemaphoreSlim sem_HandleNewImg = new SemaphoreSlim (1, 1);
			static SemaphoreSlim sem_Pop = new SemaphoreSlim (1, 1);

			static SemaphoreSlim sem_CurrentCamImg = new SemaphoreSlim (1, 1);

			/// <summary>
			/// Waits for the next image.
			/// </summary>
			/// <returns>Last input image computed</returns>
			public async Task<UIImage> Pop ()
			{
				
				try {
					await sem_CurrentCamImg.WaitAsync ();

					var img = currentCamImg;

					if (img != null) {
						currentCamImg = null;

						return img;
					}
				} finally {
					sem_CurrentCamImg.Release ();
				}


				await sem_Pop.WaitAsync ();

				try {
					await sem_CurrentCamImg.WaitAsync ();

					var img = currentCamImg;

					currentCamImg = null;

					return img;
				} finally {
					sem_CurrentCamImg.Release ();
				}
			}

			/// <summary>
			/// Processes current camera image. 
			/// </summary>
			/// <param name="captureOutput">Capture output.</param>
			/// <param name="sampleBuffer">Sample buffer.</param>
			/// <param name="connection">Connection.</param>
			public override async void DidOutputSampleBuffer (AVCaptureOutput captureOutput, CMSampleBuffer sampleBuffer, AVCaptureConnection connection)
			{
				try {
					await sem_HandleNewImg.WaitAsync ();

					var cameraImg = await ImageFromSampleBuffer (sampleBuffer);

					using (var pool = new NSAutoreleasePool ()) {
						sampleBuffer.Dispose ();
					}

					await sem_CurrentCamImg.WaitAsync ();

					currentCamImg = new UIImage (cameraImg, 1, UIImageOrientation.Right);

					if (sem_Pop.CurrentCount == 0) {
						sem_Pop.Release ();
					} 

					sem_CurrentCamImg.Release ();

				} finally {
					sem_HandleNewImg.Release ();
				}

			}

			Task<CGImage> ImageFromSampleBuffer (CMSampleBuffer sampleBuffer)
			{
				return Task.Run (() => {
					using (var pool = new NSAutoreleasePool ()) {
						// Get the CoreVideo image
						using (var pixelBuffer = sampleBuffer.GetImageBuffer () as CVPixelBuffer) {
							// Lock the base address
							pixelBuffer.Lock (CVPixelBufferLock.None);
							// Get the number of bytes per row for the pixel buffer
							var baseAddress = pixelBuffer.BaseAddress;
							var bytesPerRow = (int)pixelBuffer.BytesPerRow;
							var width = (int)pixelBuffer.Width;
							var height = (int)pixelBuffer.Height;
							var flags = CGBitmapFlags.PremultipliedFirst | CGBitmapFlags.ByteOrder32Little;
							// Create a CGImage on the RGB colorspace from the configured parameter above
							using (var cs = CGColorSpace.CreateDeviceRGB ()) {
								using (var context = new CGBitmapContext (baseAddress, width, height, 8, bytesPerRow, cs, (CGImageAlphaInfo)flags)) {
									using (CGImage cgImage = context.ToImage ()) {
										pixelBuffer.Unlock (CVPixelBufferLock.None);
										return cgImage.Clone ();
									}
								}
							}
						}
					}
				});
			}

			void TryDispose (IDisposable obj)
			{
				if (obj != null)
					obj.Dispose ();
			}
		}
	}


}


