using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VI.AOD.Segmentation.SegmentationService
{
    public interface ISegmentationService
    {
        void Initialize();

        Dictionary<string, string[]> RunSegmentation(string samplesDirectory, string outputDirectory, string cudaVisibleDevices, int batchSize, Action<string> logOutput);
    }
}
