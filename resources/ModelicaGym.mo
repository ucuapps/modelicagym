package ModelicaGym
    model CartPole "Classic cart pole simulated with an inverted pendulum"
                parameter Modelica.SIunits.Mass m_cart = 10;
                parameter Modelica.SIunits.Mass m_pole = 1;
                parameter Modelica.SIunits.Length length = 2;
                parameter Modelica.SIunits.Angle theta_0 = 85/180*pi;
                parameter Modelica.SIunits.AngularVelocity theta_dot_0 = 0.0;
                constant Real pi = Modelica.Constants.pi;
                inner Modelica.Mechanics.MultiBody.World world(gravityType=Modelica.Mechanics.MultiBody.Types.GravityTypes.
                            UniformGravity, animateWorld=false)
                                                            annotation (Placement(visible = true, transformation(origin = {-134, -48}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Joints.Prismatic prismatic(useAxisFlange=true)
                    annotation (Placement(transformation(extent={{-96,0},{-76,20}})));
                Modelica.Mechanics.Translational.Components.Damper damper1(d=0)
                    annotation (Placement(transformation(extent={{-96,14},{-76,34}})));
                Modelica.Mechanics.MultiBody.Joints.Revolute rev(n={0,0,1},useAxisFlange=true,
                    phi(fixed=true, start=theta_0),
                    w(fixed=true, start=theta_dot_0))
                                                                          annotation (Placement(visible = true, transformation(extent = {{-38, 0}, {-18, 20}}, rotation = 0)));
                Modelica.Mechanics.Rotational.Components.Damper damper(d=0)
                    annotation (Placement(visible = true, transformation(extent = {{-34, 20}, {-14, 40}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Parts.Body poleMassCenter(
                    m=m_pole,
                    r_CM={0,0,0},
                    specularCoefficient=4*world.defaultSpecularCoefficient,
                    sphereDiameter=1.5*world.defaultBodyDiameter)
                    annotation (Placement(visible = true, transformation(extent = {{30, 0}, {50, 20}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Parts.BodyShape cart(
                    shapeType="box",
                    animateSphere=true,
                    m=m_cart,
                    sphereDiameter=world.defaultBodyDiameter,
                    r={0,0,0},
                    r_CM={0,0,0})
                    annotation (Placement(visible = true, transformation(extent = {{-66, 0}, {-46, 20}}, rotation = 0)));
                Modelica.Mechanics.Translational.Sources.Force force
                    annotation (Placement(transformation(extent={{-98,34},{-78,54}})));
                Modelica.Mechanics.MultiBody.Sensors.RelativeAngles angle
                    annotation (Placement(visible = true, transformation(extent = {{-38, -30}, {-18, -10}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Sensors.RelativeVelocity velocity
                    annotation (Placement(transformation(extent={{-96,-30},{-76,-10}})));
                Modelica.Mechanics.MultiBody.Sensors.RelativePosition position
                    annotation (Placement(visible = true, transformation(extent = {{-96, -60}, {-76, -40}}, rotation = 0)));
                Modelica.Blocks.Interfaces.RealInput f
                    annotation (Placement(transformation(extent={{-190,-20},{-150,20}})));
                Modelica.Blocks.Interfaces.RealOutput x
                    annotation (Placement(visible = true, transformation(extent = {{34, 42}, {54, 62}}, rotation = 0), iconTransformation(extent = {{34, 42}, {54, 62}}, rotation = 0)));
                Modelica.Blocks.Interfaces.RealOutput x_dot
                    annotation (Placement(visible = true, transformation(extent = {{34, 24}, {54, 44}}, rotation = 0), iconTransformation(extent = {{34, 24}, {54, 44}}, rotation = 0)));
                Modelica.Blocks.Interfaces.RealOutput theta_dot
                    annotation (Placement(visible = true, transformation(extent = {{34, -58}, {54, -38}}, rotation = 0), iconTransformation(extent = {{34, -58}, {54, -38}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Sensors.RelativeAngularVelocity
angVel
                    annotation (Placement(visible = true, transformation(extent = {{-38, -60}, {-18, -40}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Parts.BodyCylinder poleCartConnection(
                    r={length/2,0,0},
                    specularCoefficient=0.7,
                    color={0,0,0},
                    diameter=0.05,
                    density=900)
                    "half of the pole length" annotation(
      Placement(visible = true, transformation(extent = {{-6, 0}, {14, 20}}, rotation = 0)));
              Modelica.Blocks.Interfaces.RealOutput theta
                    annotation (Placement(visible = true, transformation(extent = {{34, -42}, {54, -22}}, rotation = 0), iconTransformation(extent = {{34, -42}, {54, -22}}, rotation = 0)));
            equation
  connect(angVel.w_rel[3], theta_dot) annotation(
      Line(points = {{-28, -61}, {-28, -60}, {25, -60}, {25, -48}, {44, -48}}, color = {0, 0, 127}));
  connect(angle.angles[3], theta) annotation(
      Line(points = {{-28, -31}, {-28, -32}, {44, -32}}, color = {0, 0, 127}));
    connect(theta, theta) annotation(
      Line(points = {{44, -32}, {44, -32}}, color = {0, 0, 127}));
                connect(prismatic.frame_a, world.frame_b) annotation (Line(
                        points={{-96,10},{-110,10}, {-110, -48}, {-124, -48}},
                        color={95,95,95},
                        thickness=0.5));
  connect(position.r_rel[1], x) annotation(
      Line(points = {{-86, -61}, {-104, -61}, {-104, -58}, {-120, -58}, {-120, 64}, {9, 64}, {9, 52}, {44, 52}}, color = {0, 0, 127}));
  connect(velocity.v_rel[1], x_dot) annotation(
      Line(points = {{-86, -30.3333}, {-100, -30.3333}, {-100, -34}, {-114, -34}, {-114, 56}, {-3, 56}, {-3, 34}, {44, 34}}, color = {0, 0, 127}));
  connect(poleCartConnection.frame_b, poleMassCenter.frame_a) annotation(
      Line(points = {{14, 10}, {30, 10}}, color = {95, 95, 95}, thickness = 0.5));
  connect(angVel.frame_b, angle.frame_b) annotation(
      Line(points = {{-18, -50}, {-18, -20}}, color = {95, 95, 95}, thickness = 0.5));
  connect(angVel.frame_a, angle.frame_a) annotation(
      Line(points = {{-38, -50}, {-38, -20}}, color = {95, 95, 95}, thickness = 0.5));
  connect(poleCartConnection.frame_a, rev.frame_b) annotation(
      Line(points = {{-6, 10}, {-18, 10}}, color = {95, 95, 95}, thickness = 0.5));
  connect(angle.frame_a, rev.frame_a) annotation(
      Line(points = {{-38, -20}, {-38, 10}}, color = {95, 95, 95}, thickness = 0.5));
  connect(angle.frame_b, rev.frame_b) annotation(
      Line(points = {{-18, -20}, {-18, 10}}, color = {95, 95, 95}, thickness = 0.5));
  connect(cart.frame_b, rev.frame_a) annotation(
      Line(points = {{-46, 10}, {-38, 10}}, color = {95, 95, 95}, thickness = 0.5));
  connect(rev.support, damper.flange_a) annotation(
      Line(points = {{-34, 20}, {-34, 30}}));
  connect(damper.flange_b, rev.axis) annotation(
      Line(points = {{-14, 30}, {-8, 30}, {-8, 20}, {-28, 20}}));
  connect(prismatic.frame_b, cart.frame_a) annotation(
      Line(points = {{-76, 10}, {-66, 10}}, color = {95, 95, 95}, thickness = 0.5));
  connect(position.frame_a, velocity.frame_a) annotation(
      Line(points = {{-96, -50}, {-96, -20}}, color = {95, 95, 95}, thickness = 0.5));
  connect(position.frame_b, velocity.frame_b) annotation(
      Line(points = {{-76, -50}, {-76, -20}}, color = {95, 95, 95}, thickness = 0.5));
                connect(force.flange, prismatic.axis) annotation (Line(
                        points={{-78,44},{-78,16}},
                        color={0,127,0},
                        smooth=Smooth.None));
                connect(damper1.flange_a, prismatic.support) annotation (Line(
                        points={{-96,24},{-96,16},{-90,16}},
                        color={0,127,0},
                        smooth=Smooth.None));
                connect(damper1.flange_b, prismatic.axis) annotation (Line(
                        points={{-76,24},{-78,24},{-78,16}},
                        color={0,127,0},
                        smooth=Smooth.None));
  connect(velocity.frame_b, prismatic.frame_b) annotation(
      Line(points = {{-76, -20}, {-76, 10}}, color = {95, 95, 95}, thickness = 0.5, smooth = Smooth.None));
  connect(velocity.frame_a, prismatic.frame_a) annotation(
      Line(points = {{-96, -20}, {-96, 10}}, color = {95, 95, 95}, thickness = 0.5, smooth = Smooth.None));
                connect(f, force.f) annotation (Line(
                        points={{-170,0},{-136,0},{-136,44},{-100,44}},
                        color={0,0,127},
                        smooth=Smooth.None));
                annotation (
                    experiment(
                        StartTime=1,
                        StopTime=10,
                        __Dymola_Algorithm="Dassl"),
                    Diagram(coordinateSystem(
                            preserveAspectRatio=false,
                            extent={{-150,-100},{150,100}},
                            grid={2,2})),
                    Documentation(info="<html>
    
    Model of a simple cart-pole system. <br>
    It is based on the inverted pendulum formulation.
    </html>"),
                    Icon(coordinateSystem(preserveAspectRatio=true, extent={{-150,-100},{150,
                                    100}}), graphics={
                            Rectangle(
                                extent={{-150,122},{150,-120}},
                                lineColor={0,0,0},
                                fillColor={255,255,255},
                                fillPattern=FillPattern.Solid),
                            Rectangle(
                                extent={{-82,22},{82,18}},
                                lineColor={0,0,255},
                                fillPattern=FillPattern.Forward),
                            Rectangle(extent={{-44,54},{0,28}}, lineColor={0,0,0}),
                            Ellipse(
                                extent={{-40,34},{-28,22}},
                                lineColor={0,0,0},
                                fillPattern=FillPattern.Solid,
                                fillColor={255,255,255},
                                lineThickness=0.5),
                            Ellipse(
                                extent={{-16,34},{-4,22}},
                                lineColor={0,0,0},
                                fillColor={255,255,255},
                                fillPattern=FillPattern.Solid,
                                lineThickness=0.5),
                            Line(
                                points={{-18,-16},{10,-62}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Ellipse(
                                extent={{4,-56},{20,-72}},
                                lineColor={0,0,0},
                                fillColor={0,255,255},
                                fillPattern=FillPattern.Solid),
                            Ellipse(
                                extent={{-25,44},{-19,38}},
                                lineColor={0,0,0},
                                fillColor={95,95,95},
                                fillPattern=FillPattern.Solid),
                            Line(
                                points={{28,46},{4,46}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Line(
                                points={{34,40},{10,40}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Line(
                                points={{-22,40},{-18,-16}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Ellipse(
                                extent={{-20,-15},{-14,-21}},
                                lineColor={0,0,0},
                                fillColor={95,95,95},
                                fillPattern=FillPattern.Solid)}));
            end CartPole;
  
    end ModelicaGym;