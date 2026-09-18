from pathlib import Path
ROOT=Path(__file__).resolve().parent
original=ROOT.parent/'quality_capture_20260918'
script=(original/'build_run.py').read_text()
script=script.replace('#include <fstream>','#include <deque>\n#include <fstream>')
script=script.replace('Eigen::Matrix3f previousBody=Eigen::Matrix3f::Identity();','std::deque<std::pair<double,int>> recentStates;\n Eigen::Matrix3f previousBody=Eigen::Matrix3f::Identity();')
body=(original/'quality_body.inc').read_text()
body=body.replace('QualityContext& c = qualityContext();','QualityContext& c = qualityContext();\n    c.recentStates.emplace_back(c.inputTs,int(mState));\n    if(c.recentStates.size()>21) c.recentStates.pop_front();')
body=body.replace('    const double loggerMs=',(ROOT/'pair_body.inc').read_text()+'\n    const double loggerMs=')
(ROOT/'quality_body.inc').write_text(body)
(ROOT/'build_run.py').write_text(script)
(ROOT/'audit.py').write_text((original/'audit.py').read_text())
print('PREPARED_V2')
