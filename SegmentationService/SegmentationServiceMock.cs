using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VI.AOD.Segmentation.SegmentationService
{
    public class SegmentationServiceMock : ISegmentationService
    {
        public void Initialize()
        {
           
        }

        public Dictionary<string, string[]> RunSegmentation(string samplesDirectory, string outputDirectory, string cudaVisibleDevices, int batchSize, Action<string> logOutput)
        {
            var resultDictionary = new Dictionary<string, string[]>();
            List<string> aviFiles = Directory.EnumerateFiles(samplesDirectory, "*.avi").ToList();
            string cameraNumber=Path.GetFileName(samplesDirectory);
            string directoryName = Directory.GetParent(samplesDirectory).Parent.FullName;
            string segmentationResultsDir=Path.Combine(directoryName, "ADR",cameraNumber);
            List<string> segmentationResults = Directory.EnumerateFiles(segmentationResultsDir,"*.avi").ToList();

            for(int i = 0,j=0; i < segmentationResults.Count()/2; ++i,j+=2)
            {
                var imageFile=segmentationResults[j];
                var labelFile = segmentationResults[j+1];
                resultDictionary.Add(aviFiles[i],new string[] {imageFile,labelFile });
            }
            return resultDictionary;
        }
    }
}
