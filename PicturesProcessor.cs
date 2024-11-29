// Default URL for triggering event grid function in the local environment.
// http://localhost:7071/runtime/webhooks/EventGrid?functionName={functionname}

using Azure.Messaging.EventGrid;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.EventGrid;
using Microsoft.Azure.WebJobs.Extensions;
using Microsoft.Extensions.Logging;
using Azure.Identity;
using Azure.Storage.Blobs;
using System.Security.Cryptography.X509Certificates;
using Azure.Security.KeyVault.Certificates;
using Azure.Messaging.EventGrid.SystemEvents;
using Newtonsoft.Json.Linq;
using System.Threading.Tasks;
using System;
using System.IO;
using System.Text;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using Emgu.CV;
using System.Drawing;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.Util;
using Emgu.CV.UI;
using Emgu.CV.Dnn;
using Emgu.CV.Util;
using SixLabors.ImageSharp.Processing;
using Microsoft.Extensions.FileSystemGlobbing;
using ImageMagick;
using ImageMagick.Drawing;

namespace HFTM.PictureProcessor
{
    public static class PicturesProcessor
    {
        private static HaarDetectionType _detectionType = HaarDetectionType.DoCannyPruning;
        static ILogger logger = null;

        [FunctionName(nameof(PicturesProcessor))]
        public static async Task Run([EventGridTrigger] EventGridEvent eventGridEvent, [Blob("{data.url}", FileAccess.Read, Connection = "storageAccount_STORAGE")] Stream input, ILogger log)
        {
            logger = log;

            var authenticationProvider = new DefaultAzureCredential();
            logger.LogInformation("DefaultAzureCredentials created.");

            var BLOB_STORAGE_CONNECTION_STRING = Environment.GetEnvironmentVariable("storageAccount_STORAGE");

            var blobServiceClient = new BlobServiceClient(BLOB_STORAGE_CONNECTION_STRING);
            logger.LogInformation("Blob Service Client created.");

            // determine fileName of document to be uploaded
            var fileName = GetUploadedDocumentFileName(eventGridEvent);
            logger.LogInformation($"File name: {fileName}");

            // determine extension of document to be uploaded
            var extension = GetUploadedDocumentExtension(eventGridEvent);
            logger.LogInformation($"File extension: {extension}");

            // fetch new photo from storage
            var newPhoto = await GetUploadedPhotoFromStorageAsStream(eventGridEvent, blobServiceClient);

            // load into imagesharp
            var image = SixLabors.ImageSharp.Image.Load(newPhoto);

            // convert to png
            Stream outputStreamPng = new MemoryStream();
            try
            {
                var encoder = new PngEncoder();
                image.Save(outputStreamPng, encoder);
            }
            catch (Exception ex)
            {
                logger.LogError("Failed to convert image to png.");
                logger.LogError(ex.Message);
            }

            Bitmap bitmap = new Bitmap(outputStreamPng);
            var cvImg = bitmap.ToImage<Bgr, byte>();
            Image<Gray, byte> grayframe = cvImg.Convert<Gray, byte>();


            string exePath = System.Reflection.Assembly.GetExecutingAssembly().Location;
            string exeDir = System.IO.Path.GetDirectoryName(exePath);
            DirectoryInfo binDir = System.IO.Directory.GetParent(exeDir);
            string haarCascadePath = binDir.FullName + "\\haarcascade_frontalface_default.xml";

            CascadeClassifier _cascadeClassifier = new CascadeClassifier(haarCascadePath);
            
            var rectangles = _cascadeClassifier.DetectMultiScale(grayframe, 1.1, 10, new Size(20, 20));

            /*
            string yunetPath = binDir.FullName + "\\face_detection_yunet_2023mar.onnx";
            var yunetDetector = DnnInvoke.ReadNetFromONNX(yunetPath);
            yunetDetector.SetInput(cvImg);
            VectorOfMat outBlobs = new VectorOfMat(1);
            yunetDetector.Forward(outBlobs);
            var outputBlob = outBlobs[0].ToBitmap();

            var outputFacePngYunet = new MemoryStream();
            */
            if (rectangles.Length == 1)
            {
                logger.LogInformation($"Detected face.");
                logger.LogInformation($"Face rectangle coordinates: {rectangles[0].X}/{rectangles[0].Y} Size: {rectangles[0].Width}/{rectangles[0].Height}");

                // calculate cropping values
                var originalWidth = rectangles[0].Width;
                var originalHeight = rectangles[0].Height;
                var originalY = rectangles[0].Y;
                var originalX = rectangles[0].X;

                // + 70% to each height and width
                var newWidth = originalWidth + (originalWidth * 0.7);
                var newHeight = originalHeight + (originalHeight * 0.7);

                // correct x and y for increased width and height
                var totalIncreaseInWidth = newWidth - originalWidth;
                var totalIncreaseInHeight = newHeight - originalHeight;

                // we want it to be centered, but a little bit adjusted so we have more picture at the top
                var newX = originalX - (totalIncreaseInWidth * 0.5);
                var newY = originalY - (totalIncreaseInHeight * 0.6);

                // we cannot go bigger than the original picture and X/Y cannot be negative
                if (newX < 0)
                    newX = 0;
                if (newY < 0)
                    newY = 0;
                if (newX + newWidth > image.Width)
                    newWidth = originalWidth;
                if (newY + newHeight > image.Height)
                    newHeight = originalHeight;

                // crop photo using imagesharp
                var face = image.Clone(x => x.Crop(new SixLabors.ImageSharp.Rectangle((int)newX, (int)newY, (int)newWidth, (int)newHeight)));

                // save image to memory stream
                Stream outputFacePng = new MemoryStream();
                try
                {
                    var encoder = new PngEncoder();
                    face.Save(outputFacePng, encoder);
                }
                catch (Exception ex)
                {
                    logger.LogError("Failed to convert face image to png.");
                    logger.LogError(ex.Message);
                }

                // load into magick.net
                outputFacePng.Position = 0;
                MagickImage magickImage = new MagickImage(outputFacePng);

                // detect topleft most pixel color (assuming its the background)
                var color = magickImage.GetPixels()[0, 0].ToColor();

                // make that background transparent
                magickImage.ColorFuzz = new Percentage(15);
                magickImage.FloodFill(new MagickColor(0, 0, 0, 0), 1, 1); // start transparent flood fill from top left
                magickImage.FloodFill(new MagickColor(0, 0, 0, 0), (int)magickImage.Width - 1, (int)magickImage.Height - 1); // start transparent flood fill from bottom right

                // draw a new image with the "moments of red" beam
                var momentsOfRed = new MagickImage(new MagickColor(0,0,0,0), magickImage.Width, magickImage.Height);

                // create polygon in pointd
                PointD[] points = new PointD[3];
                points[0] = new PointD((int)magickImage.Width - 1, (int)magickImage.Height * 0.19);
                points[1] = new PointD((int)magickImage.Width * 0.4, (int)magickImage.Height - 1);
                points[2] = new PointD((int)magickImage.Width - 1, (int)magickImage.Height - 1);

                new Drawables()
                    .StrokeColor(new MagickColor("#E81A3B"))
                    .FillColor(new MagickColor("#E81A3B"))
                    .Polygon(points)
                    .Draw(momentsOfRed);

                // combine the two images into one
                momentsOfRed.Composite(magickImage, CompositeOperator.Atop, Channels.All);

                // save image to memory stream
                Stream outputFacePngTransparent = new MemoryStream();
                magickImage.Write(outputFacePngTransparent, MagickFormat.Png);


                await CopyPhotoToResultStorage(outputFacePngTransparent, eventGridEvent, blobServiceClient, "-adaboostffa");
                /*
                try
                {
                    outputBlob.Save(outputFacePngYunet, System.Drawing.Imaging.ImageFormat.Png);
                }
                catch (Exception ex)
                {
                    logger.LogError("Failed to convert face image to png.");
                    logger.LogError(ex.Message);
                }
                await CopyPhotoToResultStorage(outputFacePngYunet, eventGridEvent, blobServiceClient, "-yunet");
                */
                logger.LogInformation($"Saved face version.");

            }
            else if (rectangles.Length == 0)
            {
                logger.LogWarning("No faces detected.");
            }
            else
            {
                logger.LogWarning("Detected multiple faces, aborting.");
            }


        }

        private static BlobContainerClient GetBlobContainerClient(BlobServiceClient blobServiceClient, string containerName)
        {
            var blobContainerClient = blobServiceClient.GetBlobContainerClient(containerName);
            logger.LogInformation("Blob Container client created.");
            return blobContainerClient;
        }

        private static string GetUploadedDocumentExtension(EventGridEvent eventGridEvent)
        {
            var createdEvent = eventGridEvent.Data.ToObjectFromJson<StorageBlobCreatedEventData>();
            var extension = Path.GetExtension(createdEvent.Url);
            extension = extension.Replace(".", "");
            logger.LogInformation("Extension: " + extension);
            return extension;
        }

        private static string GetUploadedDocumentFileName(EventGridEvent eventGridEvent)
        {
            var url = eventGridEvent.Data.ToObjectFromJson<StorageBlobCreatedEventData>().Url;
            return url.Split("/")[url.Split("/").Length - 1];
        }

        private static string GetUploadedDocumentFileNameWithoutExtension(EventGridEvent eventGridEvent)
        {
            var fileName = GetUploadedDocumentFileName(eventGridEvent);
            return fileName.Substring(0, fileName.LastIndexOf('.'));
        }

        private static X509Certificate2 FetchKeyVaultCertificate(DefaultAzureCredential creds)
        {
            var keyVaultCertificateClient = new CertificateClient(vaultUri: new Uri(Environment.GetEnvironmentVariable("KeyVaultUrl")), credential: creds);
            logger.LogInformation("Key Vault Client created.");
            DownloadCertificateOptions downloadCertificateOptions = new DownloadCertificateOptions(Environment.GetEnvironmentVariable("KeyVaultAppRegCertificateName"));
            downloadCertificateOptions.KeyStorageFlags = X509KeyStorageFlags.EphemeralKeySet | X509KeyStorageFlags.MachineKeySet;
            var certificate = keyVaultCertificateClient.DownloadCertificate(downloadCertificateOptions);
            return certificate.Value;
        }

        private static async Task<Stream> GetUploadedPhotoFromStorageAsStream(EventGridEvent eventGridEvent, BlobServiceClient blobServiceClient)
        {
            var url = eventGridEvent.Data.ToObjectFromJson<StorageBlobCreatedEventData>().Url;
            var fileName = url.Split("/")[url.Split("/").Length - 1];
            var blobContainerClient = GetBlobContainerClient(blobServiceClient, Environment.GetEnvironmentVariable("uploadContainerName"));
            logger.LogInformation("File name: " + fileName);
            var blobClient = blobContainerClient.GetBlobClient(fileName);
            logger.LogInformation("Reading from storage: " + blobContainerClient.Uri);
            var binaryData = await blobClient.OpenReadAsync();
            return binaryData;
        }
        private static async Task CopyPhotoToResultStorage(Stream photo, EventGridEvent eventGridEvent, BlobServiceClient blobServiceClient, string suffix = "", string formatExtension = "png")
        {
            var resultContainerClient = GetBlobContainerClient(blobServiceClient, Environment.GetEnvironmentVariable("mainContainerName"));
            var name = $"Processed-{DateTime.Now.ToString("dd-MM-yyyy HH:mm:ss")}-{GetUploadedDocumentFileNameWithoutExtension(eventGridEvent)}{suffix}.{formatExtension}";
            await SavePhotoToStorage(name, resultContainerClient, photo);
        }

        private static async Task SavePhotoToStorage(string name, BlobContainerClient blobContainerClient, Stream photo)
        {
            photo.Position = 0;
            if (await DoesBlobExist(name, blobContainerClient))
            {
                var blobClient = blobContainerClient.GetBlobClient(name);
                await blobClient.UploadAsync(photo, true); // overwrite
            }
            else
            {
                await blobContainerClient.UploadBlobAsync(name, photo);
            }
        }

        private static async Task<bool> DoesBlobExist(string name, BlobContainerClient blobContainerClient)
        {
            var exists = false;
            try
            {
                var blobClient = blobContainerClient.GetBlobClient(name);
                exists = await blobClient.ExistsAsync();
            }
            catch
            {
                exists = false;
            }
            return exists;
        }


    }
}
