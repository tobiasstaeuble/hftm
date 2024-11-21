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

            // load into magick.net
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

            logger.LogInformation($"Detected {rectangles.Length} faces.");
            logger.LogInformation($"First face rectangle coordinates: {rectangles[0].X}/{rectangles[0].Y} Size: {rectangles[0].Width}/{rectangles[0].Height}");


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
    }
}
