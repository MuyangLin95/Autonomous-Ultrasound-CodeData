clearvars
for tp_Img = 1:100

ImgFoldername = ['Group_1_CA']; % choose an CA or nCA group folder to proceed
ITest = imread(['logistic_model_Validation images\',ImgFoldername,'\' ,int2str(tp_Img) '.png']);

% imtool(ITest,[]);

roidep1 = 200;
roidep2 = 500;
roidep3 = 800;

Iroi1 = ITest(roidep1:roidep2,:);
Iroi2 = ITest(roidep2+1:roidep3,:);


% ROI 1

IroiGauss1 = imgaussfilt(Iroi1,8,'Padding','symmetric');
Ibound1 = edge(IroiGauss1,'Canny',0.6); %,[0.05 0.25]

% ROI 2

IroiGauss2 = imgaussfilt(Iroi2,8,'Padding','symmetric');
Ibound2 = edge(IroiGauss2,'Canny',0.6); %,[0.05 0.25]


% Recognition
for tp = 1:1250
    AntWall(tp) = mean(find(Ibound1(:,tp),4));
    PostWall(tp) = mean(find(Ibound2(:,tp),4));
    
end

AntWall=smooth(AntWall, 50);
PostWall=smooth(PostWall, 50);

if sum(isnan(AntWall)) > 250 || sum(isnan(AntWall)) > 250
%     IResult(tp_Img) = 0;
    disp('Bad recognition')
else
    
    AntWall(isnan(AntWall)) = 0;
    PostWall(isnan(PostWall)) = 0;
    
    AntWall = smooth(AntWall(2:end-1));
    PostWall = smooth(PostWall(2:end-1));
    
    AntWallFFT = abs(fft(AntWall));
    PostWallFFT = abs(fft(PostWall));
    
    if sum(AntWallFFT(4:10)) > 2e4 || sum(PostWallFFT(4:10)) > 2e4
%         IResult(tp_Img) = 1;
    else
%         IResult(tp_Img) = 0;
    end
end
% end



%Classifier

[maximum1, index1]=max(AntWallFFT(4:40));  % anterior wall FFT peak
[maximum2, index2]=max(PostWallFFT(4:40)); % posterior wall FFT peak
if [(1<index1) & (index1<6) & (1<index2) & (index2<6)]  % heart beam range 48 bpm-108bpm (corresponding index range 1-6)
    CApossibility=1;
else
    CApossibility=0;
end

%Result=['CA possibility = ',num2str(CApossibility)] 
CAresults(tp_Img,1)=CApossibility;
end

sum(CAresults) % the result is the number of true positive cases (when the sript runs in CA folder) or false negative casues (when the sript runs in nCA folder) 